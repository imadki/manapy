import numba
import inspect
from numba import cuda
import hashlib
from mpi4py import MPI
from manapy.backends.types import FLOAT_TYPE, INT_TYPE


def get_function_hash(func):
  """Compute hash of function's source code to detect changes."""
  source = inspect.getsource(func).encode('utf-8')
  return hashlib.md5(source).hexdigest()

def get_type(s: 'str'):
  """
  return base_type and type dimension
  """
  base_type = None
  if s.startswith("float64"):
    base_type = numba.float64
  elif s.startswith("float32"):
    base_type = numba.float32
  elif s.startswith("float"):
    base_type = numba.float64 if FLOAT_TYPE == "float64" else numba.float32
  elif s.startswith("uint32"):
    base_type = numba.uint32
  elif s.startswith("uint64"):
    base_type = numba.uint64
  elif s.startswith("int32"):
    base_type = numba.int32
  elif s.startswith("int64"):
    base_type = numba.int64
  elif s.startswith("int8"):
    base_type = numba.int8
  elif s.startswith("int"):
    base_type = numba.int64 if INT_TYPE == "int64" else numba.int32

  n_dim = s.count(":")

  if base_type is None:
    raise ValueError(f"Unsupported string annotation: {s}")
  elif n_dim == 0:
    return base_type
  else:
    return numba.types.Array(base_type, n_dim, 'C')

def get_arg_types(func):
  arg_types = []
  for param in inspect.signature(func).parameters.values():
    anno = param.annotation
    if anno is inspect.Parameter.empty:
      raise ValueError(f"Parameter {param.name} lacks type annotation")
    if isinstance(anno, str):
      t = get_type(anno)
      arg_types.append(t)
    else:
      raise ValueError(f"Unsupported annotation type: {anno}")
  return tuple(arg_types)


def _compile_numba(backend: str, func, signature, parallel=False, nogil=False):
  if backend == "numba":
    return numba.jit(signature, nopython=True, fastmath=False, cache=True, parallel=parallel, nogil=nogil)(func)
  elif backend == "cuda":
    return cuda.jit(signature, fastmath=True, device=True)(func)
  else:
    raise ValueError(f"Unsupported backend: {backend}")


def compile(func, backend="numba", parallel=False, skip_on_error=False, nogil=False):
  # return func
  if backend == "python":
    return func

  # Store hash of function source to detect changes
  current_hash = get_function_hash(func)
  cached_hash = getattr(func, '_source_hash', None)

  # If function is compiled and source hasn't changed, return cached version
  if getattr(func, 'signatures', None) and current_hash == cached_hash:
    # print("already compiled =>", func)
    return func

  # Parse arguments if not cached or source changed
  if skip_on_error:
    try:
      signature = get_arg_types(func)
    except (ValueError, TypeError):
      return func  # Return uncompiled function on error
  else:
    # print("Getting signature =>", func)
    signature = get_arg_types(func)

  # Compile and store hash
  comm = MPI.COMM_WORLD
  if comm.Get_size() > 1:
    if comm.Get_rank() == 0:
      # print(f"compile function {func.__name__} with backend:", backend, "using rank ", MPI.COMM_WORLD.Get_rank())
      compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
      comm.Barrier()
    else:
      comm.Barrier()
      compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
    comm.Barrier()
  else:
    compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)

  # Attach source hash to compiled function
  compiled_func._source_hash = current_hash
  return compiled_func

"""
Compile a function once it called the first time (not immediately).
"""
class FunObj:
  def __init__(self, func, *a, **kw):
    self.target_func = func
    self.args = a
    self.kw = kw

    def com(*args):
      print("Compiling...", self.target_func.__name__)
      self.func = compile(self.target_func, *self.args, **self.kw)
      return func(*args)

    self.func = com

  def __call__(self, *args):
    return self.func(*args)