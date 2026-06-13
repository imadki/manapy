# -*- coding: utf-8 -*-
from manapy.backends.compile_fun import compile
from manapy.backends.types import FLOAT_TYPE, INT_TYPE, np_int_type, np_float_type


_BACKEND_ALIASES = {
  "cpu": "cpu", "numba": "cpu",
  "gpu": "gpu", "cuda": "gpu",
}


def get_backend(name="cpu", **kwargs):
  """
  Renvoie une instance de backend de calcul.

    name : "cpu"/"numba" -> CPUBackend (numba.njit)
           "gpu"/"cuda"  -> GPUBackend (numba.cuda.jit)
    kwargs : passes au constructeur du backend (ex. float_precision pour le GPU).

  Les imports sont paresseux : demander "cpu" ne charge jamais numba.cuda.
  """
  key = _BACKEND_ALIASES.get(str(name).lower())
  if key == "cpu":
    from manapy.backends.cpu import CPUBackend
    return CPUBackend(**kwargs)
  if key == "gpu":
    from manapy.backends.gpu import GPUBackend
    return GPUBackend(**kwargs)
  raise ValueError(f"Backend inconnu : {name!r}; attendu l'un de {sorted(set(_BACKEND_ALIASES))}")
