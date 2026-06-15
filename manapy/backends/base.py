# -*- coding: utf-8 -*-
"""
Interface commune des backends de calcul (CPU / GPU).

Idee directrice : les kernels sont ecrits une seule fois, avec des annotations de
type sous forme de chaines ('float[:]', 'int[:,:]', ...). C'est le BACKEND qui
decide comment les compiler :
  - CPU -> numba.njit  (via manapy/backends/compile_fun.py, avec sync MPI du cache)
  - GPU -> numba.cuda.jit

Les deux backends exposent la MEME interface (ci-dessous) ; seules les vraies
differences (njit vs cuda.jit, lancement sequentiel vs grille, gestion
device/stream) divergent dans les sous-classes. Les methodes device
(init_stream / to_device / synchronize) sont des no-op cote CPU pour que le code
appelant traite CPU et GPU de maniere uniforme.
"""
from abc import ABC, abstractmethod
import inspect


def map_type(annotation, float_precision, int_precision):
  """Mappe une annotation chaine vers un type concret, en isolant le suffixe
  tableau pour eviter toute corruption de sous-chaine :

    'float[:]'  -> 'float64[:]'   (si float_precision == 'float64')
    'int[:,:]'  -> 'int32[:,:]'
    'uint32[:]' -> 'int32[:]'
    'float64'   -> 'float64'      (type explicite : inchange)
  """
  if not isinstance(annotation, str):
    raise TypeError(f"Annotation de type attendue en chaine, recu : {annotation!r}")
  bracket = annotation.find("[")
  base = (annotation if bracket == -1 else annotation[:bracket]).strip()
  suffix = "" if bracket == -1 else annotation[bracket:]
  if base == "float":
    base = float_precision
  elif base in ("int", "uint32"):
    base = int_precision
  return base + suffix


class Backend(ABC):
  #: nom court du backend ("cpu", "gpu", ...)
  name = "base"

  def __init__(self, float_precision, int_precision, cache=True):
    self.float_precision = float_precision
    self.int_precision = int_precision
    self.cache = cache

  # ------------------------------------------------------------------ config
  def set_config(self, float_precision=None, int_precision=None, cache=None, **kwargs):
    if float_precision is not None:
      self.float_precision = float_precision
    if int_precision is not None:
      self.int_precision = int_precision
    if cache is not None:
      self.cache = cache
    # attributs specifiques au backend (nb_blocks, nb_threads, free, ...)
    for key, val in kwargs.items():
      if val is not None and hasattr(self, key):
        setattr(self, key, val)

  # ---------------------------------------------------------------- signatures
  def get_arg_types(self, func):
    """(return_type, (arg_types...)) en chaines, mappes a la precision du backend."""
    sig = inspect.signature(func)
    arg_types = []
    for param in sig.parameters.values():
      anno = param.annotation
      if anno is inspect.Parameter.empty:
        raise ValueError(
          f"Le parametre {param.name} de {func.__name__} n'a pas d'annotation de type")
      arg_types.append(map_type(anno, self.float_precision, self.int_precision))
    ret = sig.return_annotation
    return_type = ("void" if ret is inspect.Signature.empty
                   else map_type(ret, self.float_precision, self.int_precision))
    return return_type, tuple(arg_types)

  def build_signature(self, func):
    """Chaine de signature numba : 'ret(arg0, arg1, ...)'."""
    return_type, arg_types = self.get_arg_types(func)
    return f"{return_type}({', '.join(arg_types)})"

  # ----------------------------------------------- device / memoire (no-op CPU)
  def init_stream(self):
    """Selectionne le device et ouvre le stream (GPU). No-op cote CPU."""
    return None

  def synchronize(self):
    """Attend la fin des operations device. No-op cote CPU."""
    pass

  # ---- modele memoire (style Executor Ginkgo) : une array vit sur UN backend.
  #      L'allocation se fait via ce backend ; les transferts host<->device sont
  #      explicites (to_host / to_device). Pas d'etat double host/device.
  @abstractmethod
  def zeros(self, shape, dtype):
    """Alloue un tableau de zeros DANS la memoire de ce backend."""
    raise NotImplementedError

  @abstractmethod
  def empty(self, shape, dtype):
    """Alloue un tableau non initialise DANS la memoire de ce backend."""
    raise NotImplementedError

  @abstractmethod
  def asarray(self, data, dtype=None):
    """Place `data` (numpy/list) dans la memoire de ce backend."""
    raise NotImplementedError

  @abstractmethod
  def to_host(self, arr):
    """Renvoie une copie numpy (host) de `arr`."""
    raise NotImplementedError

  @abstractmethod
  def to_device(self, arr):
    """Garantit que `arr` est dans la memoire de ce backend (identite cote CPU)."""
    raise NotImplementedError

  @abstractmethod
  def copy(self, dst, src):
    """Copie en place le contenu de `src` (host ou backend) dans `dst` (tableau de
    ce backend). UNE seule copie O(n), uniquement a l'appel. Agnostique CPU/GPU."""
    raise NotImplementedError

  # ---------------------------------------------------------------- compilation
  @abstractmethod
  def compile_kernel(self, func, **kwargs):
    """Compile `func` (annotee en chaines) et renvoie le callable compile."""
    raise NotImplementedError

  @abstractmethod
  def make_gridstride_kernel(self, body, size_arg=None):
    """Compile un corps grid-stride `body(start, stride, *args)` pour la cible."""
    raise NotImplementedError
