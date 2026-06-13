import numba
import inspect
import os
import time
import atexit
from numba import cuda
import hashlib
from mpi4py import MPI
from manapy.backends.types import FLOAT_TYPE, INT_TYPE

_TRUE_VALUES = {"1", "true", "yes", "on"}
_TIMING_MODE = os.environ.get("MANAPY_COMPILE_TIMING", "").lower()
_TIMING_PRINT_EACH = _TIMING_MODE in _TRUE_VALUES or _TIMING_MODE == "all"
_TIMING_SUMMARY = _TIMING_MODE in _TRUE_VALUES or _TIMING_MODE in {"summary", "all"}
_TIMING_ALL_RANKS = _TIMING_MODE == "all"
_COMPILE_TIMINGS = []


def _function_name(func):
  return f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}"


def _record_compile_timing(func, backend, parallel, nogil, compile_time, elapsed_time, rank):
  entry = {
    "name": _function_name(func),
    "backend": backend,
    "parallel": parallel,
    "nogil": nogil,
    "rank": rank,
    "compile_time": compile_time,
    "elapsed_time": elapsed_time,
  }
  _COMPILE_TIMINGS.append(entry)

  if _TIMING_PRINT_EACH and (_TIMING_ALL_RANKS or rank == 0):
    print(
      "[manapy.compile] "
      f"rank={rank} {entry['name']} "
      f"compile={compile_time:.6f}s elapsed={elapsed_time:.6f}s "
      f"backend={backend} parallel={parallel}",
      flush=True,
    )


def reset_compile_timings():
  _COMPILE_TIMINGS.clear()


def get_compile_timings():
  return [entry.copy() for entry in _COMPILE_TIMINGS]


def get_compile_total_time(include_wait=False):
  key = "elapsed_time" if include_wait else "compile_time"
  return sum(entry[key] for entry in _COMPILE_TIMINGS)


def print_compile_timings(limit=None, sort=True, include_wait=True):
  entries = get_compile_timings()
  if sort:
    key = "elapsed_time" if include_wait else "compile_time"
    entries.sort(key=lambda entry: entry[key], reverse=True)
  if limit is not None:
    entries = entries[:limit]

  total_compile = get_compile_total_time(include_wait=False)
  total_elapsed = get_compile_total_time(include_wait=True)
  rank = entries[0]["rank"] if entries else MPI.COMM_WORLD.Get_rank()
  print(
    "[manapy.compile] "
    f"rank={rank} functions={len(_COMPILE_TIMINGS)} "
    f"compile_total={total_compile:.6f}s elapsed_total={total_elapsed:.6f}s",
    flush=True,
  )
  for entry in entries:
    print(
      "[manapy.compile] "
      f"rank={entry['rank']} {entry['name']} "
      f"compile={entry['compile_time']:.6f}s "
      f"elapsed={entry['elapsed_time']:.6f}s",
      flush=True,
    )


def _print_compile_timings_at_exit():
  if _TIMING_SUMMARY and _COMPILE_TIMINGS:
    rank = _COMPILE_TIMINGS[0]["rank"]
    if _TIMING_ALL_RANKS or rank == 0:
      print_compile_timings(limit=10)


atexit.register(_print_compile_timings_at_exit)


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
  rank = comm.Get_rank()
  elapsed_start = time.perf_counter()
  compile_time = 0.0
  if comm.Get_size() > 1:
    if rank == 0:
      # print(f"compile function {func.__name__} with backend:", backend, "using rank ", MPI.COMM_WORLD.Get_rank())
      compile_start = time.perf_counter()
      compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
      compile_time = time.perf_counter() - compile_start
      comm.Barrier()
    else:
      comm.Barrier()
      compile_start = time.perf_counter()
      compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
      compile_time = time.perf_counter() - compile_start
    comm.Barrier()
  else:
    compile_start = time.perf_counter()
    compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
    compile_time = time.perf_counter() - compile_start
  elapsed_time = time.perf_counter() - elapsed_start

  # Attach source hash to compiled function
  compiled_func._source_hash = current_hash
  _record_compile_timing(func, backend, parallel, nogil, compile_time, elapsed_time, rank)
  return compiled_func

"""
Compile a function once it called the first time (not immediately).
"""
class FunObj:
  def __init__(self, func, *a, **kw):
    self.target_func = func
    self.func = self._first_call
    self.args = a
    self.kw = kw

  def _first_call(self, *args):
    print("Compiling...", self.target_func.__name__)
    compiled = compile(self.target_func, *self.args, **self.kw)
    self.func = compiled  # replace for future calls
    return compiled(*args)

  def __call__(self, *args):
    return self.func(*args)