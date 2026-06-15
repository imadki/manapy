import json
import os
import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import numpy.typing as npt
from typing import Union


class GinkgoDistributedSolver(LinearSolver):
  """Fully MPI-distributed linear solver backed by Ginkgo.

  Unlike :class:`GinkgoSolver` (which gathers the whole system on rank 0), this
  solver works like PETSc: every rank assembles only its own rows and columns
  into a ``gko::experimental::distributed::Matrix`` / ``Vector``, and Ginkgo
  handles the off-process communication internally during the solve. No gather
  and no scatter.

  Requires pyGinkgo built with ``GINKGO_BUILD_MPI=ON`` (the ``distributed``
  submodule must be available).
  """

  _parameters = [
    ('scheme', 'str', 'diamond', 'fv4',
     'scheme diamond or fv4.'),
    ('reordering', 'bool', False, False,
     'unused (kept for API symmetry); distributed reordering is not supported.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, build the matrix and partition once and reuse them.'),
    ('with_mtx', 'bool', False, True,
     'If True, the matrix should be given.'),
    ('device', 'str', 'cuda', False,
     'Ginkgo executor: cpu, omp, cuda, hip or dpcpp.'),
    ('method', 'str', 'gmres', False,
     'Krylov solver: gmres, cg, bicgstab, cgs, fcg or ir.'),
    ('precond', 'str', 'none', False,
     'Distributed preconditioner: none or jacobi (block-jacobi via Schwarz).'),
    ('i_max', 'int', 1000, False,
     'The maximum number of iterations.'),
    ('eps_r', 'float', 1e-7, False,
     'The relative residual reduction factor.'),
    ('solver_args', 'dict', {}, {},
     'Raw Ginkgo solver configuration (JSON-serializable dict). Overrides '
     'method/precond/i_max/eps_r entirely.'),
    ('verbose', 'bool', False, False,
     'If True, print convergence information.'),
  ]

  _METHODS = {
    'gmres': 'solver::Gmres',
    'cg': 'solver::Cg',
    'bicgstab': 'solver::Bicgstab',
    'cgs': 'solver::Cgs',
    'fcg': 'solver::Fcg',
  }

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Intracomm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = True,
               with_mtx: bool = False,
               device: str = "cuda",
               method: str = "gmres",
               precond: str = "none",
               i_max: int = 1000,
               eps_r: float = 1e-7,
               solver_args: dict = None,
               verbose: bool = False,
               ):

    try:
      import pyGinkgo as pg
      from pyGinkgo import pyGinkgoBindings as pGB
    except ImportError:
      import sys
      print("pyGinkgo is not installed. Please install it first.")
      sys.exit(1)

    if not getattr(pg, "distributed", None) or not pg.distributed.available:
      raise RuntimeError(
        "pyGinkgo was not built with MPI support (GINKGO_BUILD_MPI=ON). "
        "The distributed solver is unavailable; use GinkgoSolver instead.")

    self.pg = pg
    self.pGB = pGB
    self.dpc = pg.distributed

    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme,
                          reordering=False,
                          solver_name=LinearSolver.SolverGinkgoDist)

    self._dim = self.domain.dim
    self.var = var
    self.domain.solver = "ginkgo_dist"
    self.reuse_mtx = reuse_mtx
    self.with_mtx = with_mtx
    self.device = device
    self.method = method
    self.precond = precond
    self.i_max = i_max
    self.eps_r = eps_r
    self.solver_args = solver_args if solver_args else {}
    self.verbose = verbose

    self._value = "double" if FLOAT_TYPE == "float64" else "float"
    self._np_value = np.float64 if FLOAT_TYPE == "float64" else np.float32

    self.executor = pg.device(device)
    # The pyGinkgo distributed wrapper accepts the mpi4py communicator directly
    # (it converts it internally), so no manual .py2f() handle is needed.

    # Global indices of the cells owned by this rank (int32: the distributed
    # binding uses int32 global indices).
    self._owned_global = np.ascontiguousarray(
      self.domain.cells.loctoglob[:self.localsize], dtype=np.int32)

    # Distributed objects (built lazily)
    self.partition = None
    self.A = None
    self.solver = None
    self.b = None
    self.x = None
    self._perm = None       # owned cell i -> local slot in the Ginkgo vector
    self._local_n = None

    # PETSc-style local RHS (size = localsize), alloue dans la memoire du backend :
    # ecrit par le kernel _get_rhs (device sous GPU) puis passe au solveur.
    self.rhs0 = self.domain.backend.zeros(self.localsize, FLOAT_TYPE)

  def _profile_time(self, label, start):
    if not (self.verbose or os.environ.get("MANAPY_GINKGO_PROFILE") == "1"):
      return
    elapsed = MPI.Wtime() - start
    elapsed = self.comm.allreduce(elapsed, op=MPI.MAX)
    if self.comm.Get_rank() == 0:
      print(f"[GinkgoDist profile] {label}: {elapsed:.6f}s", flush=True)

  # ------------------------------------------------------------------ helpers
  def _build_solver_args(self):
    if self.solver_args:
      return self.solver_args
    if self.method not in self._METHODS:
      raise ValueError(
        f"Unknown Ginkgo method '{self.method}'. Choices: "
        f"{', '.join(self._METHODS)}")
    args = {
      "type": self._METHODS[self.method],
      "criteria": [
        {"type": "Iteration", "max_iters": self.i_max},
        {"type": "ResidualNorm", "reduction_factor": self.eps_r},
      ],
    }
    if self.precond == "jacobi":
      # Block-Jacobi on the local block via a Schwarz wrapper (distributed).
      args["preconditioner"] = {
        "type": "preconditioner::Schwarz",
        "local_solver": {"type": "preconditioner::Jacobi"},
      }
    elif self.precond not in ("none", None):
      raise ValueError(
        f"Unsupported distributed preconditioner '{self.precond}'. "
        "Choices: none, jacobi.")
    return args

  def _build_partition(self):
    """Build the global owner mapping (owners[global_row] = rank) and the
    Ginkgo partition, plus the local<->global permutation."""
    size = self.comm.Get_size()
    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] build_partition: owners", flush=True)
    owners = np.empty(self.globalsize, dtype=np.int32)
    # Each rank advertises the global ids it owns; OR-reduce a per-rank stamp.
    local_owner = np.full(self.globalsize, -1, dtype=np.int32)
    local_owner[self._owned_global] = self.comm.Get_rank()
    self.comm.Allreduce(local_owner, owners, op=MPI.MAX)

    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] build_partition: ginkgo partition", flush=True)
    self.partition = self.dpc.build_partition(self.executor, owners, size)

    # Recover the Ginkgo local ordering: build a vector whose values are the
    # global indices, then read back its processor-local part.
    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] build_partition: ordering vector", flush=True)
    gidx = self.dpc.vector(self.executor, self.comm, self.partition,
                           self._owned_global,
                           self._owned_global.astype(self._np_value),
                           self.globalsize, dtype=self._value)
    local_g = np.rint(self.dpc.vector_local(gidx, dtype=self._value)).astype(np.int64)
    self._local_n = local_g.size

    # _perm[i] = local slot holding the solution of owned cell i
    pos = {int(g): k for k, g in enumerate(local_g)}
    self._perm = np.array([pos[int(g)] for g in self._owned_global],
                          dtype=np.int64)

  # ------------------------------------------------------------------ solve
  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]] = None):

    t0 = MPI.Wtime()
    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] presolve begin", flush=True)
    self.presolve(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx)
    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] presolve done", flush=True)
    self._profile_time("presolve", t0)

    # Local RHS contribution (BC) + optional user rhs, for owned cells only.
    t0 = MPI.Wtime()
    if self._gpu:
      from manapy.backends.gpu import GPUArray
      if rhs is None:
        # rhs0 was filled on the device by the GPU _get_rhs kernel: pass its
        # device buffer directly (zero-copy).
        local_rhs = GPUArray.to_device(self.rhs0)
      else:
        local_rhs = np.ascontiguousarray(self.domain.backend.to_host(self.rhs0) + rhs, dtype=self._np_value)
    else:
      local_rhs = self.rhs0 if rhs is None else (self.rhs0 + rhs)
      local_rhs = np.ascontiguousarray(local_rhs, dtype=self._np_value)

    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] rhs vector", flush=True)
    self.b = self.dpc.vector(self.executor, self.comm, self.partition,
                             self._owned_global, local_rhs, self.globalsize,
                             dtype=self._value)
    if self.x is None:
      self.x = self.dpc.vector(self.executor, self.comm, self.partition,
                               self._owned_global,
                               np.zeros(self.localsize, dtype=self._np_value),
                               self.globalsize, dtype=self._value)
    self._profile_time("vectors", t0)

    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] solve begin", flush=True)
    solver_args = json.dumps(self._build_solver_args())
    logger = None
    if self.reuse_mtx and self.method != "cg":
      if self.solver is None:
        t0 = MPI.Wtime()
        self.solver = self.pGB.solver.__getattribute__(f"config_solver_{self._value}")(
          self.executor, self.A, solver_args)
        self._profile_time("config_solver", t0)
      t0 = MPI.Wtime()
      self.solver.apply(self.b, self.x)
      self._profile_time("solver_apply", t0)
    else:
      t0 = MPI.Wtime()
      logger = self.pGB.solver.__getattribute__(f"config_solve_{self._value}")(
        self.executor, self.A, self.b, self.x, solver_args)
      self._profile_time("config_solve", t0)
    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] solve done", flush=True)

    t0 = MPI.Wtime()
    local_sol = self.dpc.vector_local(self.x, dtype=self._value)

    if ((self.verbose or os.environ.get("MANAPY_GINKGO_PROFILE") == "1")
        and logger is not None and self.comm.Get_rank() == 0):
      print(f"[GinkgoDist] {self.method}/{self.precond} "
            f"converged={logger.has_converged()} "
            f"iters={logger.get_num_iterations()}", flush=True)

    # Write each owned cell's solution directly: no scatter needed. var.cell is a
    # device array under GPU; the sliced assignment from the host `local_sol`
    # copies host->device (numba), so the GPU kernels see the updated solution.
    self.var.cell[:self.localsize] = local_sol[self._perm].astype(FLOAT_TYPE)
    self._profile_time("copy_solution", t0)

  def presolve(self, reuse_mtx=False, with_mtx=False):
    if reuse_mtx and self.A is not None:
      return

    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] update ghost", flush=True)
    self.update_ghost_values()

    if not with_mtx:
      if self.verbose:
        print(f"[GinkgoDist rank {self.comm.Get_rank()}] assembly", flush=True)
      self.assembly()
    elif not np.any(self._data):
      raise ValueError(
        "with_mtx=True but no matrix was provided: fill (row, col, data) "
        "via set_matrix(...) first, or use with_mtx=False.")

    if self.partition is None:
      self._build_partition()

    if self.verbose:
      print(f"[GinkgoDist rank {self.comm.Get_rank()}] matrix", flush=True)
    if self._gpu:
      # Pass the device buffers directly (int32 indices match the binding):
      # the assembly kernel filled them on the GPU, so this is zero-copy.
      from manapy.backends.gpu import GPUArray
      rows = GPUArray.to_device(self._row)
      cols = GPUArray.to_device(self._col)
      data = GPUArray.to_device(self._data)
    else:
      rows = np.ascontiguousarray(self._row, dtype=np.int32)
      cols = np.ascontiguousarray(self._col, dtype=np.int32)
      data = np.ascontiguousarray(self._data, dtype=self._np_value)

    # assembly_mode::communicate (inside the binding) routes contributions for
    # halo rows owned by other ranks to their owner -- like PETSc assembly.
    self.A = self.dpc.matrix(self.executor, self.comm, self.partition,
                             rows, cols, data, self.globalsize,
                             dtype=self._value)

  def set_matrix(self, row, col, data):
    """Provide the matrix triplets explicitly (with_mtx=True). Indices are
    0-based global indices."""
    self._row = np.asarray(row, dtype=np.int32)
    self._col = np.asarray(col, dtype=np.int32)
    self._data = np.asarray(data, dtype=FLOAT_TYPE)

  def view(self):
    if self.comm.Get_rank() == 0 and self.A is not None:
      print(f"[GinkgoDist] device={self.device} value={self._value} "
            f"global_size={self.globalsize} ranks={self.comm.Get_size()} "
            f"method={self.method} precond={self.precond}")

  def clear(self):
    self.partition = None
    self.A = None
    self.solver = None
    self.b = None
    self.x = None
    self._perm = None
