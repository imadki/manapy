# -*- coding: utf-8 -*-
"""
Interface commune des backends de calcul (CPU / GPU).

Idee directrice (reprise de la tentative `cuda`) : les kernels sont ecrits une
seule fois, avec des annotations de type sous forme de chaines ('float[:]',
'int[:,:]', ...). C'est le BACKEND qui decide comment les compiler :
  - CPU  -> numba.njit  (voir manapy/backends/compile_fun.py)
  - GPU  -> numba.cuda.jit

Un backend expose au minimum `compile_kernel(func, ...)` qui transforme une
fonction Python annotee en callable compile pour sa cible.
"""
from abc import ABC, abstractmethod


class Backend(ABC):
  #: nom court du backend ("cpu", "gpu", ...)
  name = "base"

  @abstractmethod
  def compile_kernel(self, func, **kwargs):
    """Compile `func` (annotee en chaines) et renvoie le callable compile."""
    raise NotImplementedError
