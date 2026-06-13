# -*- coding: utf-8 -*-
"""
Backend GPU : compile les kernels via numba.cuda.jit.

Adapte de la tentative `manapy/cuda/utils/GPU_Backend.py` (branche `cuda`),
avec trois changements :
  - precision par defaut tiree de manapy.backends.types (au lieu d'etre codee
    en dur a float32) ;
  - mapping d'annotations robuste (pas de corruption de sous-chaine du genre
    'float64' -> 'float3264') ;
  - sortie silencieuse par defaut (alignee avec la reduction du bruit MPI) ;
    activable via MANAPY_GPU_VERBOSE=1.

Note : importer ce module n'exige PAS de GPU. Seuls `init_stream()` et la
compilation/execution effective des kernels touchent le driver CUDA.
"""
import os
import inspect

from numba import cuda

import manapy.backends.types as types
from manapy.backends.base import Backend


def _verbose():
  return os.environ.get("MANAPY_GPU_VERBOSE", "0") not in ("0", "", "false", "False")


def _map_type(annotation, float_precision, int_precision):
  """
  Mappe une annotation de type (chaine) vers une signature numba.cuda.

  Decoupe la partie 'base' (avant un eventuel '[') du suffixe tableau ('[:]',
  '[:,:]', ...), mappe le token de base, puis recompose. Cela evite les
  corruptions de sous-chaine d'un simple str.replace.

    'float[:]'    -> 'float64[:]'   (si float_precision == 'float64')
    'int[:,:]'    -> 'int32[:,:]'
    'uint32[:]'   -> 'int32[:]'     (compat kernels de la branche cuda)
    'float64'     -> 'float64'      (type explicite : inchange)
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
  # sinon : type explicite (float32/float64/int32/int64/...) -> inchange

  return base + suffix


class GPUBackend(Backend):
  name = "gpu"

  def __init__(self, float_precision=None, int_precision=None, cache=True,
               nb_blocks=32, nb_threads=32, free=False):
    # GPU : float32 souvent prefere, mais on suit la precision globale par defaut.
    self.float_precision = float_precision or types.FLOAT_TYPE
    self.int_precision = int_precision or types.INT_TYPE
    self.cache = cache
    self.nb_blocks = nb_blocks
    self.nb_threads = nb_threads
    self.free = free
    self.stream = None

  # ---------------------------------------------------------------- config
  def set_config(self, float_precision=None, int_precision=None, cache=None,
                 nb_blocks=None, nb_threads=None, free=None):
    if float_precision is not None:
      self.float_precision = float_precision
    if int_precision is not None:
      self.int_precision = int_precision
    if cache is not None:
      self.cache = cache
    if nb_blocks is not None:
      self.nb_blocks = nb_blocks
    if nb_threads is not None:
      self.nb_threads = nb_threads
    if free is not None:
      self.free = free

  # ----------------------------------------------------------- device/stream
  def init_stream(self):
    """Selectionne un GPU par rang MPI et ouvre le stream par defaut."""
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    gpu_id = rank % len(cuda.gpus)
    cuda.select_device(gpu_id)
    if _verbose():
      print(f"[Rank {rank}] GPU {cuda.gpus[gpu_id].name} (id {gpu_id})")
    self.stream = cuda.default_stream()
    # Devient le backend GPU actif du process (utilise par GPUArray).
    from manapy.backends.gpu import set_active_backend
    set_active_backend(self)
    return self.stream

  # ---------------------------------------------------------------- signatures
  def get_arg_types(self, func):
    """Renvoie (return_type, (arg_types...)) en chaines, mappes a la precision."""
    sig = inspect.signature(func)
    arg_types = []
    for param in sig.parameters.values():
      anno = param.annotation
      if anno is inspect.Parameter.empty:
        raise ValueError(f"Le parametre {param.name} de {func.__name__} n'a pas d'annotation de type")
      arg_types.append(_map_type(anno, self.float_precision, self.int_precision))

    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
      return_type = "void"
    else:
      return_type = _map_type(ret, self.float_precision, self.int_precision)
    return return_type, tuple(arg_types)

  def build_signature(self, func):
    """Construit la chaine de signature cuda : 'ret(arg0, arg1, ...)'."""
    return_type, arg_types = self.get_arg_types(func)
    return f"{return_type}({', '.join(arg_types)})"

  # ---------------------------------------------------------------- compilation
  def compile_kernel(self, func, device=False):
    """Compile `func` en kernel CUDA (device=False) ou fonction device (True)."""
    signature = self.build_signature(func)
    if _verbose():
      kind = "device" if device else "kernel"
      print(f"compile {func.__name__} -> cuda {kind} : {signature}")
    return cuda.jit(signature, fastmath=True, cache=self.cache, device=device)(func)

  # ---------------------------------------------------------------- lancement
  def get_gpu_params(self, size):
    """(grid, block) pour lancer un kernel sur `size` elements."""
    if self.free:
      return (size // self.nb_threads + 1, self.nb_threads)
    return (self.nb_blocks, self.nb_threads)


  def assign(self, arr_out, value):
    if not hasattr(self, "_assign_kernel"):
      def kernel_assign(arr_out: 'float[:]', value: 'float'):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, arr_out.shape[0], stride):
          arr_out[i] = value
      self._assign_kernel = self.compile_kernel(kernel_assign)

    from manapy.backends.gpu import GPUArray
    d_out = GPUArray.to_device(arr_out)
    grid, block = self.get_gpu_params(len(arr_out))
    self._assign_kernel[grid, block, self.stream](d_out, value)
    if self.stream is not None:
      self.stream.synchronize()

  # ------------------------------------------------------ corps grid-stride unifie
  def make_gridstride_kernel(self, body, size_arg=None):
    """
    Compile un corps grid-stride `body(start, stride, *args)` pour le GPU.

    `body` est compile en device function (signature inferee), et un kernel mince
    genere dynamiquement lui fournit cuda.grid(1)/cuda.gridsize(1) puis transmet
    les arguments. `size_arg` indique comment dimensionner la grille :
      None      -> len(args[0])
      int i     -> len(args[i])
      (i,j,...) -> max(len(args[i]), ...)   (kernels a plusieurs boucles)
      callable  -> size_arg(args)
    """
    dev = cuda.jit(device=True)(body)  # lazy : pas de signature explicite

    params = list(inspect.signature(body).parameters.values())[2:]  # drop start, stride
    names = [p.name for p in params]
    src = (
      "def _gridstride_kernel(%s):\n"
      "    _dev(cuda.grid(1), cuda.gridsize(1), %s)\n"
      % (", ".join(names), ", ".join(names))
    )
    ns = {"cuda": cuda, "_dev": dev}
    exec(src, ns)  # noqa: S102  (generation controlee, noms issus de la signature)
    kfn = ns["_gridstride_kernel"]
    kfn.__annotations__ = {p.name: p.annotation for p in params}

    kernel = self.compile_kernel(kfn, device=False)
    resolve = _make_size_resolver(size_arg)

    def wrapper(*args):
      from manapy.backends.gpu import GPUArray
      d = [GPUArray.to_device(a) for a in args]
      grid, block = self.get_gpu_params(resolve(args))
      kernel[grid, block, self.stream](*d)
      if self.stream is not None:
        self.stream.synchronize()

    return wrapper


def _make_size_resolver(size_arg):
  if size_arg is None:
    return lambda args: len(args[0])
  if callable(size_arg):
    return size_arg
  if isinstance(size_arg, int):
    return lambda args: len(args[size_arg])
  idxs = tuple(size_arg)
  return lambda args: max(len(args[i]) for i in idxs)
