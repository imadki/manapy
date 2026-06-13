# -*- coding: utf-8 -*-
"""
Backend CPU : compile les kernels via numba.njit.

C'est une facade fine au-dessus de `manapy.backends.compile_fun.compile`, qui
porte deja toute la logique CPU existante (parsing des annotations en chaines
vers des types numba, et synchronisation de la compilation en MPI via
MANAPY_COMPILE_SYNC). Le but ici est uniquement d'exposer cette logique
derriere la meme interface `Backend.compile_kernel` que le GPUBackend, pour que
le code appelant puisse traiter CPU et GPU de maniere uniforme.

La precision (float/int) du CPU est globale, definie dans manapy.backends.types
et utilisee directement par compile_fun ; on l'expose ici en lecture seule pour
la symetrie avec le GPUBackend.
"""
import manapy.backends.types as types
from manapy.backends.base import Backend
from manapy.backends.compile_fun import compile as _compile


class CPUBackend(Backend):
  name = "cpu"

  def __init__(self):
    # Precision CPU : globale (types.py). Exposee pour la symetrie d'interface.
    self.float_precision = types.FLOAT_TYPE
    self.int_precision = types.INT_TYPE

  def compile_kernel(self, func, parallel=False, nogil=False, skip_on_error=False):
    """
    Compile `func` (annotee en chaines) en njit, via le moteur existant.

    Conserve la synchronisation MPI de la compilation (MANAPY_COMPILE_SYNC) et
    la detection de changement de source deja gerees par compile().
    """
    return _compile(func, backend="numba", parallel=parallel,
                    nogil=nogil, skip_on_error=skip_on_error)

  def make_gridstride_kernel(self, body, size_arg=None, parallel=False):
    """
    Compile un corps grid-stride `body(start, stride, *args)` pour le CPU.

    Cote CPU, on itere tout en une passe : le wrapper appelle le corps njit avec
    (start=0, stride=1). `size_arg` est ignore (utile seulement au GPU).
    """
    njit_body = _compile(body, backend="numba", parallel=parallel)

    def wrapper(*args):
      njit_body(0, 1, *args)

    return wrapper
