import ctypes
import os
from functools import lru_cache

from mpi4py import MPI
import numpy as np
import numpy.typing as npt

try:
  import cupy as cp
except ImportError:
  cp = None


_FALSY = ("0", "", "false", "no", "off")


def _truthy_env(name):
  """Return None when unset, else the boolean value of the env var."""
  value = os.environ.get(name)
  if value is None:
    return None
  return value.lower() not in _FALSY


@lru_cache(maxsize=1)
def mpi_cuda_support():
  """Can this MPI library read/write device pointers?

  Never probed by actually sending a device pointer: a non CUDA-aware build
  segfaults instead of raising, so there would be nothing to catch.
  """
  # 1. Explicit override wins in both directions.
  override = _truthy_env("MANAPY_GPU_AWARE_MPI")
  if override is not None:
    return override

  # 2. Ask the MPI library itself: Open MPI >= 4.0 and MPICH >= 4.3 both export
  # MPIX_Query_cuda_support. mpi4py dlopens libmpi with RTLD_LOCAL, so the
  # symbol is NOT in the global namespace -- the by-soname attempts below are
  # what actually resolve it.
  for lib_name in (None, "libmpi.so", "libmpi.so.40", "libmpi.so.12"):
    try:
      query = ctypes.CDLL(lib_name).MPIX_Query_cuda_support
    except (OSError, AttributeError):
      continue
    query.restype = ctypes.c_int
    query.argtypes = []
    return bool(query())

  # 3. MPICH family has no query symbol; env hints are all we get.
  for name in ("MPICH_GPU_SUPPORT_ENABLED", "MPIR_CVAR_ENABLE_GPU", "I_MPI_OFFLOAD"):
    if _truthy_env(name):
      return True

  return False


@lru_cache(maxsize=1)
def cuda_available():
  """cupy importable and at least one visible device."""
  if cp is None:
    return False
  try:
    return cp.cuda.runtime.getDeviceCount() > 0
  except Exception:
    return False


"""
Define for symmetrical communication.
"""
class NeighborCommunication:
  #: resolved communication modes, see :meth:`set_device`
  CPU = "cpu"
  GPU = "gpu"
  GPU_STAGED = "gpu-staged"

  def __init__(self, neighbors: 'int[:]', send_counts: 'int[:]', send_indices: 'int[:]'):
    self.size = MPI.COMM_WORLD.Get_size()
    self.rank = MPI.COMM_WORLD.Get_rank()
    self.cache = {}
    self.neighbors: npt.NDArray[np.uint32]
    self.send_counts: npt.NDArray[np.uint32]
    self.send_indices: npt.NDArray[np.uint32]
    self.recv_counts: npt.NDArray[np.uint32]
    self.recv_total: int
    self.graph_comm:  MPI.Distgraphcomm

    if self.size == 1:
      self.neighbors = np.array([0], dtype=np.uint32)
      self.send_counts = np.zeros(1, dtype=np.uint32)
      self.send_indices = np.zeros(1, dtype=np.uint32)
      self.recv_counts = np.zeros(1, dtype=np.uint32)
      self.recv_total = 1
      self.graph_comm = MPI.COMM_WORLD.Create_dist_graph_adjacent(
        sources=self.neighbors,
        destinations=self.neighbors,
        sourceweights=None, destweights=None
      )
    else:
      # Check of send_counts and send_indices are will construct.
      if np.sum(send_counts) != len(send_indices):
        raise ValueError("Mismatch: send_counts sum != send_data size")

      self.neighbors = neighbors.astype(np.uint32)
      self.send_counts = send_counts.astype(np.uint32)
      self.send_indices = send_indices.astype(np.uint32)

      # --- Create graph communicator ---
      self.graph_comm = MPI.COMM_WORLD.Create_dist_graph_adjacent(
        sources=neighbors,
        destinations=neighbors,
        sourceweights=None, destweights=None
      )

      # Creat recv_counts
      rcount = np.zeros(len(neighbors), dtype=np.uint32)
      self.graph_comm.Neighbor_alltoallv(self.send_counts, rcount)
      self.recv_counts = rcount
      self.recv_total = np.sum(rcount)

    # Ranks with no neighbor never touch the wire; keeps every path uniform.
    self._active = self.size > 1 and self.neighbors.shape[0] > 0

    # --- device state, populated by set_device("gpu") ---
    self.device = self.CPU
    self.comm_mode = self.CPU
    self._d_send_indices = None
    self._host_bufs = {}
    # (request, send buffer) kept alive until the caller waits on the request
    self._pending = []

  # ------------------------------------------------------------------ device

  def set_device(self, device, strict=False):
    """Select where halo exchanges happen. Returns the resolved mode.

    ``device`` is ``"cpu"`` or ``"gpu"`` (``"cuda"``/``"host"`` accepted).

    Resolved modes:
      ``"cpu"``         host buffers, host MPI;
      ``"gpu"``         device buffers handed straight to a CUDA-aware MPI;
      ``"gpu-staged"``  device gather, host MPI, device scatter back.

    COLLECTIVE: every rank must call this, and all ranks agree on the result
    via an allreduce. Neighbor_alltoallv is collective, so a rank on the
    device path mixed with a rank on the staged path would hang.

    Must not be called while non-blocking exchanges are in flight. With
    ``strict=True`` a request for ``"gpu"`` raises instead of degrading to
    ``"gpu-staged"``.
    """
    device = str(device).lower()
    if device in ("cpu", "host"):
      device = self.CPU
    elif device in ("gpu", "cuda"):
      device = self.GPU
    else:
      raise ValueError(f"Unknown device: {device!r} (expected 'cpu' or 'gpu')")

    self._prune_pending()
    if self._pending:
      raise RuntimeError(
        f"set_device() called with {len(self._pending)} exchange(s) in flight; "
        "wait on them first"
      )

    if device == self.CPU:
      self._release_device_state()
      self.device = self.CPU
      self.comm_mode = self.CPU
      return self.comm_mode

    if not cuda_available():
      if strict:
        raise RuntimeError(
          "set_device('gpu') requires cupy and a visible CUDA device. "
          "Install a build matching your CUDA toolkit, e.g. "
          "`pip install cupy-cuda12x`."
        )
      self._release_device_state()
      self.device = self.CPU
      self.comm_mode = self.CPU
      return self.comm_mode

    # Local capability, then global agreement: the weakest rank sets the mode.
    capable = 1 if mpi_cuda_support() else 0
    if self.size > 1:
      capable = self.graph_comm.allreduce(capable, op=MPI.MIN)

    if not capable and strict:
      raise RuntimeError(
        "set_device('gpu', strict=True): MPI is not CUDA-aware on every rank. "
        "Set MANAPY_GPU_AWARE_MPI=1 to override the detection."
      )

    self.device = self.GPU
    self.comm_mode = self.GPU if capable else self.GPU_STAGED
    self._build_device_state()
    return self.comm_mode

  def device_report(self):
    """Per-condition diagnosis, for logging a run's configuration."""
    return {
      "device": self.device,
      "comm_mode": self.comm_mode,
      "cupy_available": cp is not None,
      "cuda_device_visible": cuda_available(),
      "mpi_cuda_support": mpi_cuda_support(),
      "gpu_aware_override": _truthy_env("MANAPY_GPU_AWARE_MPI"),
      "active": self._active,
    }

  def _build_device_state(self):
    # Device-resident indices: indexing a device array with a host index array
    # would force a host->device copy of the indices on every exchange.
    self._d_send_indices = cp.asarray(self.send_indices.astype(np.int32))
    self._host_bufs = {}

  def _release_device_state(self):
    self._d_send_indices = None
    self._host_bufs = {}

  # ------------------------------------------------------------- collectives

  def exchange(self, data, recv_buffer=None):
    if not self._active:
      # No neighbors: nothing is received. Honor the caller's buffer (left
      # unchanged) and never return a copy of the local data, which would be
      # mistaken for data received from neighbors.
      if recv_buffer is not None:
        return recv_buffer
      xp = self._array_module(data)
      if data.ndim == 2:
        return xp.empty((0, data.shape[1]), dtype=data.dtype)
      return xp.empty(0, dtype=data.dtype)

    self._prune_pending()

    xp = self._array_module(data)
    mpi_type = self._mpi_type(data)
    data = xp.ascontiguousarray(data)
    block_size = self._block_size(data)
    send_counts, send_displs, recv_counts, recv_displs = self._counts(data.ndim, block_size)

    # A host array is exchanged on the host whatever the configured device.
    if xp is np or self.comm_mode == self.CPU:
      send_data = data[self.send_indices]
      recv_data = recv_buffer
      if recv_data is None:
        recv_data = self._alloc_recv(np, data.dtype, block_size)
      self.graph_comm.Neighbor_alltoallv(
        (send_data, (send_counts, send_displs), mpi_type),
        (recv_data, (recv_counts, recv_displs), mpi_type)
      )
      return recv_data

    if self.comm_mode == self.GPU_STAGED:
      return self._exchange_staged(data, recv_buffer, block_size, mpi_type,
                                   (send_counts, send_displs, recv_counts, recv_displs))

    # --- device buffers straight into MPI ---
    d_send = data[self._d_send_indices]
    d_recv = recv_buffer
    if d_recv is None:
      d_recv = self._alloc_recv(cp, data.dtype, block_size)
    # MPI knows nothing about the CUDA stream: the gather must have landed
    # before MPI reads the send buffer.
    cp.cuda.get_current_stream().synchronize()
    self.graph_comm.Neighbor_alltoallv(
      (d_send, (send_counts, send_displs), mpi_type),
      (d_recv, (recv_counts, recv_displs), mpi_type)
    )
    return d_recv

  def immediate_exchange(self, data, recv_buffer):
    if not self._active:
      # No neighbors: nothing to exchange. Return a null request so callers can
      # still call .Wait()/Waitall() uniformly (recv_buffer is left unchanged).
      return MPI.REQUEST_NULL

    self._prune_pending()

    xp = self._array_module(data)
    mpi_type = self._mpi_type(data)
    data = xp.ascontiguousarray(data)
    block_size = self._block_size(data)
    send_counts, send_displs, recv_counts, recv_displs = self._counts(data.ndim, block_size)

    if xp is np or self.comm_mode == self.CPU:
      send_data = data[self.send_indices]
      request = self.graph_comm.Ineighbor_alltoallv(
        (send_data, (send_counts, send_displs), mpi_type),
        (recv_buffer, (recv_counts, recv_displs), mpi_type)
      )
      self._pending.append((request, send_data))
      return request

    if self.comm_mode == self.GPU_STAGED:
      # Staged mode cannot overlap: the host->device scatter may only run after
      # completion, and callers use the static MPI.Request.Waitall(), which
      # accepts real requests only -- a Python wrapper carrying the scatter
      # would not survive it. Exchange synchronously, report a null request.
      self._exchange_staged(data, recv_buffer, block_size, mpi_type,
                            (send_counts, send_displs, recv_counts, recv_displs))
      return MPI.REQUEST_NULL

    # --- non-blocking with device buffers ---
    d_send = data[self._d_send_indices]
    cp.cuda.get_current_stream().synchronize()
    request = self.graph_comm.Ineighbor_alltoallv(
      (d_send, (send_counts, send_displs), mpi_type),
      (recv_buffer, (recv_counts, recv_displs), mpi_type)
    )
    # d_send is a fresh temporary; without this reference Python may free the
    # device memory while MPI is still reading it.
    self._pending.append((request, d_send))
    return request

  def _exchange_staged(self, data, recv_buffer, block_size, mpi_type, counts):
    """Device gather -> host MPI -> device scatter.

    Only the halo-sized buffers cross the PCIe bus, never the whole field.
    """
    send_counts, send_displs, recv_counts, recv_displs = counts
    nsend = self.send_indices.shape[0]

    d_send = data[self._d_send_indices]
    h_send = self._pinned("send", nsend, block_size, data.dtype)
    d_send.get(out=h_send)

    h_recv = self._pinned("recv", int(self.recv_total), block_size, data.dtype)
    self.graph_comm.Neighbor_alltoallv(
      (h_send, (send_counts, send_displs), mpi_type),
      (h_recv, (recv_counts, recv_displs), mpi_type)
    )

    if recv_buffer is None:
      return cp.asarray(h_recv)
    if isinstance(recv_buffer, np.ndarray):
      recv_buffer[...] = h_recv
    else:
      recv_buffer.set(h_recv)
    return recv_buffer

  # ----------------------------------------------------------------- helpers

  @staticmethod
  def _array_module(data):
    if cp is not None and isinstance(data, cp.ndarray):
      return cp
    return np

  @staticmethod
  def _mpi_type(data):
    mpi_type = MPI._typedict.get(data.dtype.char)
    if mpi_type is None:
      raise TypeError(f"Unsupported dtype {data.dtype}")
    return mpi_type

  @staticmethod
  def _block_size(data):
    if data.ndim == 1:
      return 1
    if data.ndim == 2:
      return data.shape[1]
    raise ValueError("Only 1D or 2D arrays are supported")

  def _counts(self, ndim, block_size):
    """Counts and displacements in scalar elements. Host metadata: identical
    for every mode, so the cache is shared."""
    key = (ndim, block_size)
    if key not in self.cache:
      send_counts = self.send_counts.astype(np.int32) * block_size
      recv_counts = self.recv_counts.astype(np.int32) * block_size

      send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0).astype(np.int32)
      recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0).astype(np.int32)

      self.cache[key] = (send_counts, send_displs, recv_counts, recv_displs)
    return self.cache[key]

  def _alloc_recv(self, xp, dtype, block_size):
    if block_size == 1:
      return xp.empty(self.recv_total, dtype=dtype)
    return xp.empty((self.recv_total, block_size), dtype=dtype)

  def _pinned(self, kind, rows, block_size, dtype):
    """Page-locked host staging buffer, reused across calls.

    Pinned memory roughly doubles the achievable D2H/H2D bandwidth over
    pageable memory.
    """
    dtype = np.dtype(dtype)
    shape = (rows,) if block_size == 1 else (rows, block_size)
    key = (kind, shape, dtype.str)
    buf = self._host_bufs.get(key)
    if buf is None:
      count = int(np.prod(shape)) if rows else 0
      mem = cp.cuda.alloc_pinned_memory(max(count, 1) * dtype.itemsize)
      buf = np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)
      self._host_bufs[key] = buf
    return buf

  def _prune_pending(self):
    """Drop send buffers whose request the caller has already waited on.

    mpi4py nulls the handle in place on Wait/Waitall, so this needs no MPI
    call and cannot touch a freed request.
    """
    if self._pending:
      self._pending = [(r, b) for r, b in self._pending if r != MPI.REQUEST_NULL]
