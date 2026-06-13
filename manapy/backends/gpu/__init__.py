# -*- coding: utf-8 -*-
from manapy.backends.gpu.gpu_backend import GPUBackend

# Backend GPU "actif" du process : partage par GPUArray et les usines de kernels
# pour retrouver le stream/la config courante sans le passer partout.
_active_backend = None


def set_active_backend(backend):
  global _active_backend
  _active_backend = backend
  return backend


def get_active_backend():
  """Renvoie le GPUBackend actif (en cree un par defaut au premier appel)."""
  global _active_backend
  if _active_backend is None:
    _active_backend = GPUBackend()
  return _active_backend


from manapy.backends.gpu.gpu_array import GPUArray  # noqa: E402  (depend de get_active_backend)
