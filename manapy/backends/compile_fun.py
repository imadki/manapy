import os
import time
import numba
import numba.core.config
import numpy as np
import inspect
import hashlib
from mpi4py import MPI


# =============================================================================
# Emplacement du cache numba : node-local (pas NFS).
#
# Mesure (advection2d.py, 112 rangs) : sur NFS le cache numba est INUTILISABLE a
# l'echelle. Quelle que soit la synchro, les 100+ rangs qui se partagent les
# memes fichiers cache soit CRASHENT (OSError 116 Stale file handle), soit
# n'arrivent pas a recharger (chaque run recompile ~tout). Un repertoire LOCAL
# au noeud (coherent, rapide) est la seule chose qui marche -> tout se charge.
#
# On pointe donc numba vers un repertoire local, une seule fois, au PREMIER
# compile() (pas a l'import). Respecte un NUMBA_CACHE_DIR explicite ; desactivable
# avec MANAPY_LOCAL_CACHE=0.
# =============================================================================
_LOCAL_CACHE_READY = False


def _ensure_local_cache_dir():
  global _LOCAL_CACHE_READY
  if _LOCAL_CACHE_READY:
    return
  _LOCAL_CACHE_READY = True
  if os.environ.get("NUMBA_CACHE_DIR") or os.environ.get("MANAPY_LOCAL_CACHE", "1") == "0":
    return  # l'utilisateur choisit lui-meme, ou desactive : on ne touche a rien
  user = os.environ.get("USER") or "user"
  for base in ("/dev/shm", os.environ.get("TMPDIR") or "/tmp"):
    d = os.path.join(base, f"manapy-numba-{user}")
    try:
      os.makedirs(d, exist_ok=True)
      numba.core.config.CACHE_DIR = d
      return
    except OSError:
      continue


# =============================================================================
# Court-circuit du scan numba `entry_points` (groupe numba_extensions).
#
# Au tout premier jit, numba scanne les entry points de TOUS les paquets
# installes (importlib.metadata.entry_points) pour decouvrir d'eventuelles
# extensions. Mesure : ~0.2 s solo mais ~3 s a 56 rangs/noeud (contention I/O
# NFS sur ~300 dist-info) -> ce cout gonfle l'init du rang le plus lent, sur
# lequel tous les autres attendent au 1er collectif MPI. Aucune extension
# `numba_extensions` n'etant enregistree, le scan ne trouve jamais rien : on
# marque l'init deja faite AVANT le 1er jit -> init_all() devient un no-op.
# Reactivable avec MANAPY_NUMBA_ENTRYPOINTS=1.
# =============================================================================
_ENTRYPOINTS_TUNED = False


def _skip_numba_entrypoints():
  global _ENTRYPOINTS_TUNED
  if _ENTRYPOINTS_TUNED:
    return
  _ENTRYPOINTS_TUNED = True
  if os.environ.get("MANAPY_NUMBA_ENTRYPOINTS", "0") == "1":
    return
  try:
    import numba.core.entrypoints as _ep
    _ep._already_initialized = True
  except Exception:
    pass


# =============================================================================
# Warmup JIT : init one-time numba/LLVM du process, payee en parallele.
#
# Le premier numba.jit d'un process paie toute l'init LLVM/typing (~0.7 s solo,
# ~14 s a 56 rangs/noeud). Sans warmup, cette init se paie DANS le verrou de la
# 1ere fonction compilee (le rang qui tient le slot la paie pendant que les
# autres attendent, puis eux la paient a leur tour -> deux inits en serie).
# Ici : compilation triviale cache=False (aucun verrou) -> tous les rangs
# s'initialisent EN MEME TEMPS. (Une variante "node-serial" a ete mesuree PLUS
# LENTE a 112 procs -> abandonnee.)
# =============================================================================
_JIT_WARMED = False


def _warmup_jit():
  global _JIT_WARMED
  if _JIT_WARMED:
    return
  _JIT_WARMED = True

  def _warm(x):
    return x + 1

  numba.jit("int64(int64)", nopython=True, cache=False)(_warm)


# =============================================================================
# Strategie de synchronisation de la compilation MPI (variable MANAPY_COMPILE_SYNC).
#
# En MPI, tous les rangs compilent les memes fonctions en meme temps et numba
# (cache=True) ecrit le resultat sur disque ; sans precaution ils se marchent
# dessus sur le fichier cache au 1er run.
#   "mpi_shared_lock" (defaut) : verrou MPI shared-memory par noeud. Le 1er rang
#                du noeud qui reserve le slot compile ; les autres attendent puis
#                lisent le cache. Strategie validee a l'echelle.
#   "current" : barrieres MPI par fonction (rang 0 compile, barriere, les autres
#                compilent, barriere). Historique, conserve pour le cas GPU+MPI.
# =============================================================================
COMPILE_SYNC = os.environ.get("MANAPY_COMPILE_SYNC", "mpi_shared_lock")


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
    # Sauvegarde du cache disque desactivable pour diagnostic (MANAPY_NUMBA_CACHE=0)
    # -> chaque rang recompile en memoire (pas de load), revele le cout memoire brut.
    use_cache = os.environ.get("MANAPY_NUMBA_CACHE", "1") != "0"
    return numba.jit(signature, nopython=True, fastmath=False, cache=use_cache, parallel=parallel, nogil=nogil)(func)
  elif backend == "cuda":
    from numba import cuda  # import paresseux : evite le scan CUDA sur noeud CPU
    return cuda.jit(signature, fastmath=True, device=True)(func)
  else:
    raise ValueError(f"Unsupported backend: {backend}")


def _local_compile(func, signature, backend, parallel, nogil, current_hash):
  """Pure local compilation (no MPI). Attaches the source hash."""
  compiled_func = _compile_numba(backend, func, signature, parallel=parallel, nogil=nogil)
  compiled_func._source_hash = current_hash
  return compiled_func


# -----------------------------------------------------------------------------
# Strategie "mpi_shared_lock" : lock MPI shared-memory par noeud (defaut).
# Un tableau de verrous en memoire partagee du noeud ; chaque fonction est
# hachee vers un slot. Le 1er rang qui reserve le slot compile, les autres
# attendent la liberation puis lisent le cache.
# -----------------------------------------------------------------------------
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
    win.Lock(0, MPI.LOCK_EXCLUSIVE)
    try:
      locks.fill(0)
      win.Sync()
    finally:
      win.Unlock(0)
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
# Strategie "current" : barrieres MPI par fonction (historique, cas GPU+MPI).
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


_STRATEGIES = {
  "mpi_shared_lock": _sync_mpi_shared_lock,
  "current": _sync_current,
}


def compile(func, backend="numba", parallel=False, skip_on_error=False, nogil=False):
  if backend == "python":
    return func
  _ensure_local_cache_dir()
  _skip_numba_entrypoints()
  _warmup_jit()

  # Store hash of function source to detect changes
  current_hash = get_function_hash(func)
  cached_hash = getattr(func, '_source_hash', None)

  # If function is compiled and source hasn't changed, return cached version
  if getattr(func, 'signatures', None) and current_hash == cached_hash:
    return func

  # Parse arguments if not cached or source changed
  if skip_on_error:
    try:
      signature = get_arg_types(func)
    except (ValueError, TypeError):
      return func  # Return uncompiled function on error
  else:
    signature = get_arg_types(func)

  # Dispatch to the selected synchronization strategy
  if COMPILE_SYNC not in _STRATEGIES:
    raise ValueError(f"Unknown MANAPY_COMPILE_SYNC={COMPILE_SYNC!r}; "
                     f"expected one of {sorted(_STRATEGIES)}")
  return _STRATEGIES[COMPILE_SYNC](func, signature, backend, parallel, nogil, current_hash)


def compile_no_cache(func, backend="numba", parallel=False, nogil=False):
  """Compile without numba's on-disk cache.

  Use for kernels whose compiled code depends on a *global* that is rebound at
  runtime (e.g. the convective kernel that inlines the currently selected flux):
  numba's disk cache keys on the kernel source/signature only, so a cached entry
  would otherwise be reused across different flux bindings. cache=False forces a
  fresh, correct compilation each process; no disk file means no MPI cache race.
  """
  if backend == "python":
    return func
  _ensure_local_cache_dir()
  _skip_numba_entrypoints()
  _warmup_jit()
  current_hash = get_function_hash(func)
  if getattr(func, 'signatures', None) and current_hash == getattr(func, '_source_hash', None):
    return func
  signature = get_arg_types(func)
  compiled = numba.jit(signature, nopython=True, fastmath=False, cache=False,
                       parallel=parallel, nogil=nogil)(func)
  compiled._source_hash = current_hash
  return compiled


class FunObj:
  """Compile a function the first time it is called (not immediately)."""
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
