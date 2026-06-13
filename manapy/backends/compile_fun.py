import os
import tempfile
import numba
import inspect
from numba import cuda
import hashlib
from mpi4py import MPI
from manapy.backends.types import FLOAT_TYPE, INT_TYPE


# =============================================================================
# Choix de la stratégie de synchronisation de la compilation (cache numba).
#
# Pourquoi : en MPI, tous les rangs compilent les mêmes fonctions en même temps
# et numba (cache=True) écrit le résultat sur disque. Sans précaution, les rangs
# se marchent dessus sur le fichier cache -> corruption / crash au premier run.
#
# Stratégies disponibles (variable d'env MANAPY_COMPILE_SYNC) :
#   "current"  : comportement historique. Barrières MPI PAR FONCTION
#                (rang 0 compile, barrière, les autres compilent, barrière).
#                Simple, mais ~2 barrières x nb_fonctions au démarrage.
#   "warmup"   : option 1. compile() est LOCAL (aucun MPI) et se contente
#                d'enregistrer la fonction ; il faut appeler warmup() une fois
#                après les imports -> UNE seule paire de barrières au total.
#   "filelock" : option 3. Verrou fichier (lib `filelock`) autour de la
#                compilation. Découplé de MPI -> aucun risque de deadlock,
#                marche tant que le système de fichiers est partagé.
#   "per_node" : option 4. Sous-communicateur par nœud (COMM_TYPE_SHARED) :
#                seul le rang local 0 de chaque nœud compile, barrière
#                node-locale, les autres lisent le cache. Évite la barrière
#                globale et la recompilation redondante inter-nœuds.
# =============================================================================
COMPILE_SYNC = os.environ.get("MANAPY_COMPILE_SYNC", "current")


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
# Stratégie "filelock" (option 3) : verrou fichier, sans barrière MPI.
# Le premier process à prendre le verrou compile et écrit le cache ; les autres
# attendent le verrou puis compilent (cache chaud -> simple chargement).
# -----------------------------------------------------------------------------
def _lock_dir():
  d = os.path.join(tempfile.gettempdir(), "manapy_compile_locks")
  os.makedirs(d, exist_ok=True)
  return d

def _sync_filelock(func, signature, backend, parallel, nogil, current_hash):
  try:
    from filelock import FileLock
  except ImportError as e:
    raise ImportError("MANAPY_COMPILE_SYNC=filelock requires the 'filelock' package") from e
  lock_name = f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}.lock"
  lock_path = os.path.join(_lock_dir(), lock_name.replace(os.sep, "_"))
  with FileLock(lock_path):
    return _local_compile(func, signature, backend, parallel, nogil, current_hash)


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
# Stratégie "warmup" (option 1) : compilation différée + UNE paire de barrières.
# compile() enregistre la fonction et renvoie un proxy ; warmup() compile tout.
#
# IMPORTANT : appeler warmup() UNE FOIS après avoir importé les modules *_compute,
# avant le premier run. Si warmup() n'est pas appelé, chaque proxy compile
# localement à son premier appel (fallback, sans barrière).
#
# Limite connue : un proxy ne peut PAS être passé comme argument à une autre
# fonction numba (njit) ; les noyaux qui s'appellent entre eux doivent référencer
# les fonctions privées `_xxx`, pas les noms publics compilés.
# -----------------------------------------------------------------------------
_WARMUP_REGISTRY = []

class _DeferredKernel:
  def __init__(self, func, signature, backend, parallel, nogil, current_hash):
    self._spec = (func, signature, backend, parallel, nogil, current_hash)
    self._source_hash = current_hash
    self.func = None
    _WARMUP_REGISTRY.append(self)

  def _compile(self):
    if self.func is None:
      func, signature, backend, parallel, nogil, h = self._spec
      self.func = _local_compile(func, signature, backend, parallel, nogil, h)
    return self.func

  @property
  def signatures(self):
    return getattr(self.func, 'signatures', None)

  def __call__(self, *args, **kwargs):
    if self.func is None:
      self._compile()  # fallback local si warmup() n'a pas été appelé
    return self.func(*args, **kwargs)

def _sync_warmup(func, signature, backend, parallel, nogil, current_hash):
  return _DeferredKernel(func, signature, backend, parallel, nogil, current_hash)

def warmup():
  """Compile toutes les fonctions enregistrées par la stratégie 'warmup',
  avec une seule paire de barrières MPI (rang 0 d'abord, puis les autres)."""
  comm = MPI.COMM_WORLD
  if comm.Get_size() > 1:
    if comm.Get_rank() == 0:
      for k in _WARMUP_REGISTRY:
        k._compile()
      comm.Barrier()
    else:
      comm.Barrier()
      for k in _WARMUP_REGISTRY:
        k._compile()
    comm.Barrier()
  else:
    for k in _WARMUP_REGISTRY:
      k._compile()


# -----------------------------------------------------------------------------
# Stratégie "lazy" (option 5) : compilation AU PREMIER APPEL.
# Ne compile QUE les fonctions réellement appelées -> dans un run 2D, les noyaux
# 3D ne sont jamais compilés (et inversement). C'est ce qui évite de compiler
# tout le 2D+3D à l'import.
#
# En MPI (size > 1), le premier appel compile sous un verrou fichier (filelock)
# pour éviter la course au cache numba. En série, compilation directe.
#
# Limite connue : comme pour 'warmup', un proxy ne peut PAS être passé comme
# argument à une autre fonction numba (les noyaux s'appellent via les `_xxx`).
# -----------------------------------------------------------------------------
class _LazyKernel:
  def __init__(self, func, signature, backend, parallel, nogil, current_hash):
    self._spec = (func, signature, backend, parallel, nogil, current_hash)
    self._source_hash = current_hash
    self.func = None

  def _compile(self):
    func, signature, backend, parallel, nogil, h = self._spec
    if MPI.COMM_WORLD.Get_size() > 1:
      from filelock import FileLock
      lock_name = f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}.lock"
      lock_path = os.path.join(_lock_dir(), lock_name.replace(os.sep, "_"))
      with FileLock(lock_path):
        self.func = _local_compile(func, signature, backend, parallel, nogil, h)
    else:
      self.func = _local_compile(func, signature, backend, parallel, nogil, h)
    return self.func

  @property
  def signatures(self):
    return getattr(self.func, 'signatures', None)

  def __call__(self, *args, **kwargs):
    if self.func is None:
      self._compile()
    return self.func(*args, **kwargs)

def _sync_lazy(func, signature, backend, parallel, nogil, current_hash):
  return _LazyKernel(func, signature, backend, parallel, nogil, current_hash)


_STRATEGIES = {
  "current": _sync_current,
  "warmup": _sync_warmup,
  "filelock": _sync_filelock,
  "per_node": _sync_per_node,
  "lazy": _sync_lazy,
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

  def _first_call(self, *args):
    print("Compiling...", self.target_func.__name__)
    compiled = compile(self.target_func, *self.args, **self.kw)
    self.func = compiled  # replace for future calls
    return compiled(*args)

  def __call__(self, *args):
    return self.func(*args)
