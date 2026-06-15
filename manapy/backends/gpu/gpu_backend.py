# -*- coding: utf-8 -*-
"""
Backend GPU : compile les kernels via numba.cuda.jit.

Meme interface que le CPUBackend (cf. manapy/backends/base.py). Les vraies
differences ("cas majeurs") :
  - compile_kernel  -> cuda.jit(signature, cache=..., device=...) ;
  - make_gridstride -> compile le corps en device function + un kernel mince qui
    fournit cuda.grid(1)/gridsize(1) ;
  - init_stream / to_device / synchronize / assign gerent le device et le stream.

Cache GPU : le wrapper grid-stride est genere dynamiquement, mais on le compile
avec le VRAI fichier source du corps (inspect.getfile) et un nom unique, pour que
numba dispose d'un locator de cache (un `exec` nu donnerait co_filename '<string>'
-> non cachable). Le cache fonctionne donc des deux cotes.

Importer ce module n'exige PAS de GPU ; seuls init_stream() et la compilation/
execution effective touchent le driver CUDA.
"""
import os
import inspect

import numpy as np
from numba import cuda

import manapy.backends.types as types
from manapy.backends.base import Backend


def _verbose():
  return os.environ.get("MANAPY_GPU_VERBOSE", "0") not in ("0", "", "false", "False")


def _make_size_resolver(size_arg):
  """Comment dimensionner la grille a partir des args du wrapper :
    None      -> len(args[0])
    int i     -> len(args[i])
    (i,j,...) -> max(len(args[i]), ...)
    callable  -> size_arg(args)
  """
  if size_arg is None:
    return lambda args: len(args[0])
  if callable(size_arg):
    return size_arg
  if isinstance(size_arg, int):
    return lambda args: len(args[size_arg])
  idxs = tuple(size_arg)
  return lambda args: max(len(args[i]) for i in idxs)


class GPUBackend(Backend):
  name = "gpu"

  def __init__(self, float_precision=None, int_precision=None, cache=True,
               nb_blocks=32, nb_threads=32, free=False):
    super().__init__(float_precision or types.FLOAT_TYPE,
                     int_precision or types.INT_TYPE, cache)
    self.nb_blocks = nb_blocks
    self.nb_threads = nb_threads
    self.free = free
    self.stream = None

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
    from manapy.backends.gpu import set_active_backend
    set_active_backend(self)
    return self.stream

  def to_device(self, arr):
    from manapy.backends.gpu import GPUArray
    return GPUArray.to_device(arr)

  def synchronize(self):
    if self.stream is not None:
      self.stream.synchronize()

  # ------------------------------------------------------------ modele memoire
  # La memoire du backend GPU est la memoire device : numba DeviceNDArray.
  def zeros(self, shape, dtype):
    return cuda.to_device(np.zeros(shape, dtype=dtype), stream=self.stream)

  def empty(self, shape, dtype):
    return cuda.device_array(shape, dtype=dtype, stream=self.stream)

  def asarray(self, data, dtype=None):
    return cuda.to_device(np.asarray(data, dtype=dtype), stream=self.stream)

  def to_host(self, arr):
    if isinstance(arr, np.ndarray):
      return arr
    host = arr.copy_to_host(stream=self.stream)
    self.synchronize()
    return host

  def to_device(self, arr):
    if isinstance(arr, np.ndarray):
      return cuda.to_device(arr, stream=self.stream)
    return arr  # deja un DeviceNDArray

  def copy(self, dst, src):
    # UNE copie : numba accepte un src host (upload) ou device (device->device).
    dst.copy_to_device(src, stream=self.stream)

  # ---------------------------------------------------------------- compilation
  def compile_kernel(self, func, device=False, cache=None):
    """cuda.jit. `cache=None` -> self.cache ; passer cache=False pour une fonction
    sans fichier source (que numba ne peut pas cacher)."""
    signature = self.build_signature(func)
    if _verbose():
      kind = "device" if device else "kernel"
      print(f"compile {func.__name__} -> cuda {kind} : {signature}")
    use_cache = self.cache if cache is None else cache
    return cuda.jit(signature, fastmath=True, cache=use_cache, device=device)(func)

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
    # pas de synchronize : meme stream que le kernel qui suit (ordonne).

  # ------------------------------------------------------ corps grid-stride
  def make_gridstride_kernel(self, body, size_arg=None):
    """Compile un corps grid-stride `body(start, stride, *args)` pour le GPU :
    le corps devient une device function, et un kernel mince lui fournit
    cuda.grid(1)/cuda.gridsize(1). Le wrapper est genere avec le VRAI fichier du
    corps + un nom unique -> cachable (pas de '<string>')."""
    dev = cuda.jit(device=True, cache=self.cache)(body)

    params = list(inspect.signature(body).parameters.values())[2:]  # drop start, stride
    names = [p.name for p in params]
    fname = f"_gridstride_{body.__name__}"
    src = (f"def {fname}({', '.join(names)}):\n"
           f"    _dev(cuda.grid(1), cuda.gridsize(1), {', '.join(names)})\n")
    ns = {"cuda": cuda, "_dev": dev}
    # Compiler avec le fichier source reel du corps (et non '<string>') donne a
    # numba un locator de cache ; le nom unique evite toute collision de cle.
    exec(compile(src, inspect.getfile(body), "exec"), ns)
    kfn = ns[fname]
    kfn.__annotations__ = {p.name: p.annotation for p in params}
    kfn.__qualname__ = fname

    kernel = self.compile_kernel(kfn, device=False)
    resolve = _make_size_resolver(size_arg)

    def wrapper(*args):
      from manapy.backends.gpu import GPUArray
      d = [GPUArray.to_device(a) for a in args]
      grid, block = self.get_gpu_params(resolve(args))
      kernel[grid, block, self.stream](*d)
      # Pas de synchronize ici : les kernels s'enchainent sur le meme stream
      # (donc ordonnes). La synchro n'a lieu qu'aux lectures host (to_host, dt).

    return wrapper
