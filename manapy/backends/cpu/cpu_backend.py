# -*- coding: utf-8 -*-
"""
Backend CPU : compile les kernels via numba.njit.

Meme interface que le GPUBackend (cf. manapy/backends/base.py). Les seules vraies
differences ("cas majeurs") :
  - compile_kernel  -> numba.njit (via compile_fun : sync MPI du cache + cache=True) ;
  - make_gridstride -> appelle le corps njit en une passe (start=0, stride=1) ;
  - les methodes device (init_stream/to_device/synchronize) sont les no-op herites
    de Backend.

Le cache CPU fonctionne nativement : le corps est une vraie fonction de fichier,
donc numba.njit(cache=True) (dans compile_fun) peut l'ecrire sur disque.
"""
import numpy as np

import manapy.backends.types as types
from manapy.backends.base import Backend
from manapy.backends.compile_fun import compile as _compile


class CPUBackend(Backend):
  name = "cpu"

  def __init__(self, float_precision=None, int_precision=None, cache=True, **kwargs):
    super().__init__(float_precision or types.FLOAT_TYPE,
                     int_precision or types.INT_TYPE, cache)

  # ---------------------------------------------------------------- compilation
  def compile_kernel(self, func, device=False, cache=None,
                     parallel=False, nogil=False, skip_on_error=False):
    """njit via le moteur existant (parsing des annotations chaines + sync MPI du
    cache). `device`/`cache` sont acceptes pour la symetrie d'interface ;
    compile_fun gere deja cache=True."""
    return _compile(func, backend="numba", parallel=parallel,
                    nogil=nogil, skip_on_error=skip_on_error)

  def make_gridstride_kernel(self, body, size_arg=None, parallel=False):
    """Cote CPU on itere tout en une passe : le wrapper appelle le corps njit
    avec (start=0, stride=1). `size_arg` est ignore (utile seulement au GPU)."""
    njit_body = _compile(body, backend="numba", parallel=parallel)

    def wrapper(*args):
      njit_body(0, 1, *args)

    return wrapper

  # ----------------------------------------------------- equivalents "device"
  def get_gpu_params(self, size):
    """Inutilise cote CPU ; present pour la symetrie d'interface."""
    return (1, 1)

  def assign(self, arr_out, value):
    arr_out[:] = value

  # ------------------------------------------------------------ modele memoire
  # Cote CPU la memoire du backend EST la memoire host : numpy partout.
  def zeros(self, shape, dtype):
    return np.zeros(shape, dtype=dtype)

  def empty(self, shape, dtype):
    return np.empty(shape, dtype=dtype)

  def asarray(self, data, dtype=None):
    return np.asarray(data, dtype=dtype)

  def to_host(self, arr):
    return np.asarray(arr)

  def to_device(self, arr):
    return arr

  def copy(self, dst, src):
    dst[:] = src
