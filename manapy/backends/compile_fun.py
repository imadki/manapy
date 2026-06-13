import os
import tempfile
import time
import numba
import numpy as np
import inspect
from numba import cuda
import hashlib
from mpi4py import MPI
from manapy.backends.types import FLOAT_TYPE, INT_TYPE


# =============================================================================
# Choix de la strategie de synchronisation de la compilation (cache numba).
#
# Pourquoi : en MPI, tous les rangs compilent les memes fonctions en meme temps
# et numba (cache=True) ecrit le resultat sur disque. Sans precaution, les rangs
# peuvent se marcher dessus sur le fichier cache au premier run.
#
# Strategies disponibles (variable d'env MANAPY_COMPILE_SYNC) :
#   "current"  : comportement historique. Barrieres MPI par fonction
#                (rang 0 compile, barriere, les autres compilent, barriere).
#   "per_node" : sous-communicateur par noeud (COMM_TYPE_SHARED) :
#                seul le rang local 0 de chaque noeud compile d'abord, puis
#                les autres rangs du meme noeud lisent le cache.
#   "claim"    : premier process arrive, premier process compile. Un lock
#                atomique par fonction empeche deux rangs de compiler la meme
#                fonction en meme temps ; les autres attendent puis lisent le cache.
#   "mpi_shared_lock" : verrou MPI shared-memory par noeud. Le premier rang du
#                noeud qui reserve le slot compile ; les autres attendent puis
#                lisent le cache.
# =============================================================================
COMPILE_SYNC = os.environ.get("MANAPY_COMPILE_SYNC", "mpi_shared_lock")


def get_function_hash(func):
  """Compute hash of function's source code to detect changes."""
  # print(func)
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


def _local_compile(func, signature, backend, parallel, nogil, current_hash):
  """Pure local compilation (no MPI). Attaches the source hash."""
  compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
  compiled_func._source_hash = current_hash
  return compiled_func


def _compile_lock_dir():
  lock_dir = os.environ.get("MANAPY_COMPILE_LOCK_DIR")
  if lock_dir is None:
    lock_dir = os.path.join(tempfile.gettempdir(), "manapy_compile_locks")
  os.makedirs(lock_dir, exist_ok=True)
  return lock_dir


def _compile_lock_path(func):
  name = f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}"
  safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
  return os.path.join(_compile_lock_dir(), safe_name + ".lock")


def _wait_for_compile_lock(lock_path):
  timeout = float(os.environ.get("MANAPY_COMPILE_LOCK_TIMEOUT", "600"))
  poll = float(os.environ.get("MANAPY_COMPILE_LOCK_POLL", "0.05"))
  start = time.monotonic()
  while os.path.exists(lock_path):
    if time.monotonic() - start > timeout:
      raise TimeoutError(f"Timed out waiting for numba compile lock: {lock_path}")
    time.sleep(poll)


_MPI_SHARED_LOCK_STATE = None


def _mpi_shared_lock_state():
  global _MPI_SHARED_LOCK_STATE
  if _MPI_SHARED_LOCK_STATE is not None:
    return _MPI_SHARED_LOCK_STATE

  node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
  nlocks = int(os.environ.get("MANAPY_COMPILE_SHARED_LOCKS", "4096"))
  dtype = np.dtype("i")
  nbytes = nlocks * dtype.itemsize if node_comm.Get_rank() == 0 else 0
  win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=node_comm)
  buf, _ = win.Shared_query(0)
  locks = np.ndarray(buffer=buf, dtype=dtype, shape=(nlocks,))

  if node_comm.Get_rank() == 0:
    locks.fill(0)
    win.Sync()
  node_comm.Barrier()

  _MPI_SHARED_LOCK_STATE = node_comm, win, locks
  return _MPI_SHARED_LOCK_STATE


def _mpi_shared_lock_slot(func):
  name = f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}"
  digest = hashlib.md5(name.encode("utf-8")).digest()
  _, _, locks = _mpi_shared_lock_state()
  return int.from_bytes(digest[:8], "little") % len(locks)


def _mpi_shared_lock_try_claim(slot):
  _, win, locks = _mpi_shared_lock_state()
  token = MPI.COMM_WORLD.Get_rank() + 1
  win.Lock(0, MPI.LOCK_EXCLUSIVE)
  try:
    win.Sync()
    if locks[slot] == 0:
      locks[slot] = token
      win.Sync()
      return True
    return False
  finally:
    win.Unlock(0)


def _mpi_shared_lock_wait(slot):
  _, win, locks = _mpi_shared_lock_state()
  timeout = float(os.environ.get("MANAPY_COMPILE_LOCK_TIMEOUT", "600"))
  poll = float(os.environ.get("MANAPY_COMPILE_LOCK_POLL", "0.05"))
  start = time.monotonic()

  while True:
    win.Lock(0, MPI.LOCK_SHARED)
    try:
      win.Sync()
      if locks[slot] == 0:
        return
    finally:
      win.Unlock(0)

    if time.monotonic() - start > timeout:
      raise TimeoutError(f"Timed out waiting for MPI shared compile lock slot {slot}")
    time.sleep(poll)


def _mpi_shared_lock_release(slot):
  _, win, locks = _mpi_shared_lock_state()
  win.Lock(0, MPI.LOCK_EXCLUSIVE)
  try:
    locks[slot] = 0
    win.Sync()
  finally:
    win.Unlock(0)


# -----------------------------------------------------------------------------
# Stratégie "current" : barrières MPI par fonction (comportement historique).
# -----------------------------------------------------------------------------
def _sync_current(func, signature, backend, parallel, nogil, current_hash):
  comm = MPI.COMM_WORLD
  if comm.Get_size() > 1:
    if comm.Get_rank() == 0:
      compiled_func = _local_compile(func, signature, backend, parallel, nogil, current_hash)
      comm.Barrier()
    else:
      comm.Barrier()
      compiled_func = _local_compile(func, signature, backend, parallel, nogil, current_hash)
    comm.Barrier()
  else:
    compiled_func = _local_compile(func, signature, backend, parallel, nogil, current_hash)
  return compiled_func


# -----------------------------------------------------------------------------
# Stratégie "per_node" (option 4) : un compilateur par nœud.
# Sous-communicateur partagé (COMM_TYPE_SHARED) créé une seule fois et réutilisé.
# -----------------------------------------------------------------------------
_NODE_COMM = None

def _node_comm():
  global _NODE_COMM
  if _NODE_COMM is None:
    _NODE_COMM = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
  return _NODE_COMM

def _sync_per_node(func, signature, backend, parallel, nogil, current_hash):
  if MPI.COMM_WORLD.Get_size() == 1:
    return _local_compile(func, signature, backend, parallel, nogil, current_hash)
  ncomm = _node_comm()
  if ncomm.Get_rank() == 0:
    compiled_func = _local_compile(func, signature, backend, parallel, nogil, current_hash)
    ncomm.Barrier()
  else:
    ncomm.Barrier()
    compiled_func = _local_compile(func, signature, backend, parallel, nogil, current_hash)
  ncomm.Barrier()
  return compiled_func


# -----------------------------------------------------------------------------
# Strategie "claim" : premier arrive, premier servi par fonction.
# -----------------------------------------------------------------------------
def _sync_claim(func, signature, backend, parallel, nogil, current_hash):
  lock_path = _compile_lock_path(func)
  while True:
    try:
      fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
      _wait_for_compile_lock(lock_path)
      return _local_compile(func, signature, backend, parallel, nogil, current_hash)

    try:
      with os.fdopen(fd, "w") as f:
        comm = MPI.COMM_WORLD
        f.write(f"rank={comm.Get_rank()} pid={os.getpid()}\n")
      return _local_compile(func, signature, backend, parallel, nogil, current_hash)
    finally:
      try:
        os.unlink(lock_path)
      except FileNotFoundError:
        pass

# -----------------------------------------------------------------------------
# Strategie "mpi_shared_lock" : lock MPI shared-memory par noeud.
# -----------------------------------------------------------------------------
def _sync_mpi_shared_lock(func, signature, backend, parallel, nogil, current_hash):
  slot = _mpi_shared_lock_slot(func)
  if _mpi_shared_lock_try_claim(slot):
    try:
      return _local_compile(func, signature, backend, parallel, nogil, current_hash)
    finally:
      _mpi_shared_lock_release(slot)

  _mpi_shared_lock_wait(slot)
  return _local_compile(func, signature, backend, parallel, nogil, current_hash)


# -----------------------------------------------------------------------------
_STRATEGIES = {
  "current": _sync_current,
  "per_node": _sync_per_node,
  "claim": _sync_claim,
  "mpi_shared_lock": _sync_mpi_shared_lock,
}


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

  # Dispatch to the selected synchronization strategy
  if COMPILE_SYNC not in _STRATEGIES:
    raise ValueError(f"Unknown MANAPY_COMPILE_SYNC={COMPILE_SYNC!r}; "
                     f"expected one of {sorted(_STRATEGIES)}")
  return _STRATEGIES[COMPILE_SYNC](func, signature, backend, parallel, nogil, current_hash)


"""
Compile a function once it called the first time (not immediately).
"""
class FunObj:
  def __init__(self, func, *a, **kw):
    self.target_func = func
    self.func = self._first_call
    self.args = a
    self.kw = kw

  def _first_call(self, *args, **kwargs):
    compiled = compile(self.target_func, *self.args, **self.kw)
    self.func = compiled  # replace for future calls
    return compiled(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)
