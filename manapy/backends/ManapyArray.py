"""
Dual-buffer host/device array with lazy, on-demand synchronization.

Design
------
Two buffers (numpy + cupy) and two validity bits. At least one bit is always
true. Every access declares an intent:

    r()   read-only    -> sync in if stale, other side stays valid
    rw()  read-write   -> sync in if stale, other side invalidated
    w()   write-only   -> NO transfer (you overwrite everything), other invalidated

State transitions (np_ok, cp_ok):

    cpu_r   : (0,1) -> (1,1)   D2H copy      | (1,*) -> unchanged
    cpu_rw  : (0,1) -> (1,0)   D2H copy      | (1,*) -> (1,0)
    cpu_w   :  any  -> (1,0)   no copy
    (gpu_* is symmetric)

Buffers are allocated once and reused: syncs copy in place via
`cp.ndarray.get(out=)` / `cp.ndarray.set()`, so steady-state has zero
allocation churn.
"""

from __future__ import annotations

import os
from enum import IntEnum

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    HAS_CUPY = False

# Opt-in guardrails. Off by default so the hot path stays branch-free-ish.
DEBUG_SYNC = bool(int(os.environ.get("MANAPY_DEBUG_SYNC", "0")))


class Device(IntEnum):
    CPU = 0
    CUDA = 1


def _require_cupy():
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available; CUDA arrays cannot be used.")


def as_device(device):
    """Normalise 'cpu' / 'cuda' / 'gpu' / 0 / 1 / Device -> Device."""
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        key = device.strip().lower()
        if key == "cpu":
            return Device.CPU
        if key in ("cuda", "gpu"):
            return Device.CUDA
    elif device == 0:
        return Device.CPU
    elif device == 1:
        return Device.CUDA
    raise ValueError(f"Unknown device: {device!r}")


def _instance_attr_names(obj):
    """Instance attributes only: no properties, no class-level constants.

    Using dir() here would call every property getter on the object, which
    can be expensive, can have side effects, and returns values that are
    recomputed on each access and therefore cannot be converted anyway.
    """
    names = list(getattr(obj, "__dict__", {}).keys())
    seen = set(names)
    for klass in type(obj).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots or ():
            if slot not in seen:
                seen.add(slot)
                names.append(slot)
    return names


class ManapyArray:
    # __slots__ removes the instance __dict__ lookup on every access.
    __slots__ = ("_np", "_cp", "_np_ok", "_cp_ok", "_shape", "_dtype")

    # ---------------------------------------------------------------- ctor

    def __init__(self, data, device=Device.CPU):
        self._np = None
        self._cp = None
        self._np_ok = False
        self._cp_ok = False

        if device == Device.CPU:
            if not isinstance(data, np.ndarray):
                raise TypeError("CPU array must be a numpy.ndarray")
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            self._np = data
            self._np_ok = True
        elif device == Device.CUDA:
            _require_cupy()
            if not isinstance(data, cp.ndarray):
                raise TypeError("CUDA array must be a cupy.ndarray")
            if not data.flags.c_contiguous:
                data = cp.ascontiguousarray(data)
            self._cp = data
            self._cp_ok = True
        else:
            raise ValueError(f"Unknown device: {device!r}")

        ref = self._np if self._np is not None else self._cp
        self._shape = ref.shape
        self._dtype = ref.dtype

    @classmethod
    def zeros(cls, shape, dtype=np.float64, device=Device.CPU):
        if device == Device.CPU:
            return cls(np.zeros(shape, dtype=dtype), Device.CPU)
        _require_cupy()
        return cls(cp.zeros(shape, dtype=dtype), Device.CUDA)

    @classmethod
    def ones(cls, shape, dtype=np.float64, device=Device.CPU):
        if device == Device.CPU:
            return cls(np.ones(shape, dtype=dtype), Device.CPU)
        _require_cupy()
        return cls(cp.ones(shape, dtype=dtype), Device.CUDA)

    @classmethod
    def full(cls, shape, fill_value, dtype=np.float64, device=Device.CPU):
        if device == Device.CPU:
            return cls(np.full(shape, fill_value, dtype=dtype), Device.CPU)
        _require_cupy()
        return cls(cp.full(shape, fill_value, dtype=dtype), Device.CUDA)

    @classmethod
    def array(cls, obj, dtype=None, device=Device.CPU, copy=True):
        """Build a ManapyArray from any array-like.

        Accepts lists, tuples, numpy arrays, cupy arrays or another
        ManapyArray, and transfers across the host/device boundary if the
        source does not already live on `device`.

        Copies by default, like `numpy.array`. This matters here: a
        ManapyArray must *own* its buffer, because an aliased input that the
        caller keeps mutating would write past the validity bits and
        reintroduce exactly the stale-data class of bug this design removes.
        Pass `copy=False` only when you are deliberately handing over
        ownership of a contiguous buffer that already sits on `device`.
        """
        if isinstance(obj, ManapyArray):
            src = obj.gpu_r() if device == Device.CUDA else obj.cpu_r()
            return cls.array(src, dtype=dtype, device=device, copy=True)

        if device == Device.CPU:
            if HAS_CUPY and isinstance(obj, cp.ndarray):
                data = cp.asnumpy(obj)                       # D2H, fresh buffer
                if dtype is not None:
                    data = data.astype(dtype, copy=False)
                data = np.ascontiguousarray(data)
            elif copy:
                data = np.array(obj, dtype=dtype, order="C")
            else:
                data = np.ascontiguousarray(obj, dtype=dtype)
            return cls(data, Device.CPU)

        if device == Device.CUDA:
            _require_cupy()
            if isinstance(obj, cp.ndarray):
                if copy:
                    data = cp.array(obj, dtype=dtype, order="C")
                else:
                    data = cp.ascontiguousarray(obj, dtype=dtype)
            else:
                # H2D always allocates a new device buffer, so `copy` is moot.
                host = np.ascontiguousarray(obj, dtype=dtype)
                data = cp.asarray(host)
            return cls(data, Device.CUDA)

        raise ValueError(f"Unknown device: {device!r}")

    @classmethod
    def empty_like(cls, other, device=None):
        device = other.resident_device if device is None else device
        return cls.zeros(other._shape, other._dtype, device)

    # ------------------------------------------------------- CPU accessors

    def cpu_r(self):
        """Host buffer for reading. Device copy stays valid."""
        if self._np_ok:
            return self._np
        self._d2h()
        return self._np

    def cpu_rw(self):
        """Host buffer for read-modify-write. Device copy is invalidated."""
        if not self._np_ok:
            self._d2h()
        elif DEBUG_SYNC:
            self._np.setflags(write=True)
        self._cp_ok = False
        return self._np

    def cpu_w(self):
        """Host buffer you will fully overwrite. No transfer is performed."""
        if self._np is None:
            self._np = np.empty(self._shape, dtype=self._dtype)
        elif DEBUG_SYNC:
            self._np.setflags(write=True)
        self._np_ok = True
        self._cp_ok = False
        return self._np

    # ------------------------------------------------------- GPU accessors

    def gpu_r(self):
        """Device buffer for reading. Host copy stays valid."""
        if self._cp_ok:
            return self._cp
        self._h2d()
        return self._cp

    def gpu_rw(self):
        """Device buffer for read-modify-write. Host copy is invalidated."""
        if not self._cp_ok:
            self._h2d()
        self._np_ok = False
        if DEBUG_SYNC and self._np is not None:
            self._np.setflags(write=False)
        return self._cp

    def gpu_w(self):
        """Device buffer you will fully overwrite. No transfer is performed."""
        if self._cp is None:
            _require_cupy()
            self._cp = cp.empty(self._shape, dtype=self._dtype)
        self._cp_ok = True
        self._np_ok = False
        if DEBUG_SYNC and self._np is not None:
            self._np.setflags(write=False)
        return self._cp

    # ------------------------------------------------------------ transfers

    def _d2h(self):
        if not self._cp_ok:
            raise RuntimeError("ManapyArray has no valid copy (internal bug)")
        if self._np is None:
            self._np = np.empty(self._shape, dtype=self._dtype)
        elif DEBUG_SYNC:
            self._np.setflags(write=True)
        self._cp.get(out=self._np)          # in-place, no allocation
        self._np_ok = True

    def _h2d(self):
        if not self._np_ok:
            raise RuntimeError("ManapyArray has no valid copy (internal bug)")
        _require_cupy()
        if self._cp is None:
            self._cp = cp.empty(self._shape, dtype=self._dtype)
        self._cp.set(self._np)              # in-place, no allocation
        self._cp_ok = True

    # ------------------------------------------------------------- utility

    def sync(self):
        """Make both copies valid. Use before handing buffers to foreign code."""
        if not self._np_ok:
            self._d2h()
        elif not self._cp_ok:
            self._h2d()
        return self

    def mark_modified(self, device):
        """Escape hatch: you wrote through a raw pointer we handed out."""
        if device == Device.CPU:
            self._np_ok, self._cp_ok = True, False
        elif device == Device.CUDA:
            self._np_ok, self._cp_ok = False, True
        else:
            raise ValueError(f"Unknown device: {device!r}")

    @property
    def resident_device(self):
        """Where the data is valid, preferring GPU when both are."""
        return Device.CUDA if self._cp_ok else Device.CPU

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self._shape[0]

    def __repr__(self):
        return (f"ManapyArray(shape={self._shape}, dtype={self._dtype}, "
                f"cpu_ok={self._np_ok}, gpu_ok={self._cp_ok})")

    # --------------------------------------------------------------- conversion

    @staticmethod
    def convert_all_tables(obj, device, strict=True, copy=True, skip=()):
        """Remplace in-place tous les attributs ndarray de `obj` par des ManapyArray.

        Ne touche qu'aux attributs d'instance (`__dict__` / `__slots__`) : les
        properties sont recalculees a chaque acces, donc les convertir n'a
        aucun sens et les lire peut couter cher.

        strict : leve une erreur si un attribut ndarray n'a pas pu etre
                 converti. Mettre a False uniquement si vous savez pourquoi
                 la conversion echoue -- un tableau laisse en ndarray brut
                 se manifestera bien plus tard sous forme d'AttributeError
                 au fond d'un kernel, ou pire, ne se manifestera pas du tout.
        copy   : voir ManapyArray.array().
        skip   : noms d'attributs a ignorer.

        Retourne la liste des attributs convertis.
        """
        device = as_device(device)
        if device == Device.CUDA:
            _require_cupy()

        converted, failed = [], []
        for name in _instance_attr_names(obj):
            if name in skip:
                continue
            try:
                value = getattr(obj, name)
            except AttributeError:      # slot declare mais non initialise
                continue
            if not isinstance(value, np.ndarray):
                continue
            try:
                setattr(obj, name, ManapyArray.array(value, device=device,
                                                     copy=copy))
                converted.append(name)
            except Exception as exc:    # attribut en lecture seule, dtype objet...
                failed.append((name, exc))

        if failed:
            detail = ", ".join(f"{n} ({type(e).__name__}: {e})" for n, e in failed)
            msg = (f"{type(obj).__name__}: {len(failed)} ndarray attribute(s) "
                   f"left unconverted: {detail}")
            if strict:
                raise RuntimeError(msg)
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return converted

    @staticmethod
    def convert_to_manapy_array(list_obj, device, strict=True, copy=True, skip=()):
        """Applique convert_all_tables() a chaque objet de `list_obj`."""
        return [ManapyArray.convert_all_tables(item, device, strict=strict,
                                               copy=copy, skip=skip)
                for item in list_obj]


# --------------------------------------------------------------------------
# Usage: hoist the accessor OUT of the loop, never call it per element.
# --------------------------------------------------------------------------
#
#   def compute_flux_cpu(w, flux, mesh):
#       w_h    = w.cpu_r()        # one branch, once per kernel
#       flux_h = flux.cpu_w()     # fully overwritten -> no transfer
#       _flux_kernel_cpu(w_h, flux_h, mesh.faceid.cpu_r())
#
#   def compute_flux_gpu(w, flux, mesh):
#       w_d    = w.gpu_r()
#       flux_d = flux.gpu_w()
#       _flux_kernel_gpu[grid, block](w_d, flux_d, mesh.faceid.gpu_r())