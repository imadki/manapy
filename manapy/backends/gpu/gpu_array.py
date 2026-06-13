# -*- coding: utf-8 -*-
"""
GPUArray : tableau GPU (adapte de la branche `cuda`).

Sous-classe de np.ndarray qui sert de "poignee" cote hote vers une copie sur
device. Le tableau hote reste accessible (c'est un ndarray), et sa copie GPU est
memorisee paresseusement dans l'attribut `__cuda__` au premier `to_device`.

Usage :
  GPUArray.convert_to_gpu_array([var, domain, domain.cells, domain.faces, ...])
    -> remplace tous les attributs ndarray de ces objets par des GPUArray.
  Les usines de kernels GPU appellent GPUArray.to_device(arg) sur chaque
  argument : scalaires laisses tels quels, GPUArray uploadee une seule fois.

Le stream utilise est celui du GPUBackend actif (manapy.backends.gpu).
"""
import numpy as np
from numba import cuda

_SCALARS = (int, float, np.int32, np.int64, np.float32, np.float64)


def _stream():
  from manapy.backends.gpu import get_active_backend
  return get_active_backend().stream


class GPUArray(np.ndarray):
  def __new__(cls, input_array):
    # Scalaires et GPUArray deja construites : rien a envelopper.
    if isinstance(input_array, (*_SCALARS, GPUArray)):
      return input_array
    obj = np.asarray(input_array).view(cls)
    obj.type_shape = f"GPUArray<{obj.dtype}, {obj.shape}>"
    return obj

  # --------------------------------------------------------------- transferts
  def to_host(self):
    """Recopie la version device dans la vue hote et la renvoie."""
    if not hasattr(self, "__on_device__"):
      return np.asarray(self)
    stream = _stream()
    dev = getattr(self, "__cuda__")
    dev.copy_to_host(np.asarray(self), stream=stream)
    if stream is not None:
      stream.synchronize()
    return np.asarray(self)

  @staticmethod
  def to_device(arr):
    """Upload vers le GPU ; cache seulement les GPUArray."""
    if isinstance(arr, _SCALARS):
      return arr
    stream = _stream()
    if not isinstance(arr, GPUArray):
      return cuda.to_device(arr, stream=stream)
    if not hasattr(arr, "__on_device__"):
      d_arr = cuda.to_device(arr, stream=stream)
      setattr(arr, "__on_device__", True)
      setattr(arr, "__cuda__", d_arr)
      return d_arr
    return getattr(arr, "__cuda__")

  def sync_with(self, side):
    """Synchronise hote<->device. side='cpu' pousse vers GPU, 'cuda' rapatrie."""
    stream = _stream()
    if hasattr(self, "__on_device__"):
      dev = getattr(self, "__cuda__")
      if side == "cpu":
        cuda.to_device(np.asarray(self), to=dev, stream=stream)
      elif side == "cuda":
        dev.copy_to_host(np.asarray(self), stream=stream)
      else:
        raise ValueError("side doit etre 'cpu' ou 'cuda'")
      return
    if side == "cpu":
      GPUArray.to_device(self)
      return
    raise RuntimeError("tableau pas encore sur device")

  # --------------------------------------------------------------- conversion
  @staticmethod
  def convert_all_tables(obj):
    """Remplace in-place tous les attributs ndarray de `obj` par des GPUArray."""
    names = []
    for attr_name in dir(obj):
      try:
        if isinstance(getattr(obj, attr_name), np.ndarray):
          names.append(attr_name)
      except Exception:
        pass
    for attr_name in names:
      try:
        setattr(obj, attr_name, GPUArray(getattr(obj, attr_name)))
      except Exception:
        # certains attributs ndarray sont des properties en lecture seule
        pass

  @staticmethod
  def convert_to_gpu_array(list_obj):
    for item in list_obj:
      GPUArray.convert_all_tables(item)
