import json
import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import numpy.typing as npt
from typing import Union
from scipy.sparse import csr_matrix


class GinkgoSolver(LinearSolver):
  """Linear solver backed by Ginkgo (pyGinkgo bindings).

  The current pyGinkgo build is single-node (GINKGO_BUILD_MPI=OFF), so the
  solve happens on a single device (CPU/OMP or CUDA). The mesh is assembled in
  a distributed way like the other solvers, then the global (row, col, data)
  triplets are gathered on rank 0, the system is solved with Ginkgo on the
  requested device, and the solution is scattered back into ``var.cell`` --
  exactly the centralized strategy used by MUMPSSolver.
  """

  _parameters = [
    ('scheme', 'str', 'diamond', 'fv4',
     'scheme diamond or fv4.'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, build the matrix and the solver once and reuse them.'),
    ('reuse_ij', 'bool', True, False,
     'If True, gather the (row, col) structure once and only refresh the '
     'numerical values on subsequent calls.'),
    ('with_mtx', 'bool', False, True,
     'If True, the matrix should be given.'),
    ('device', 'str', 'cuda', False,
     'Ginkgo executor: cpu, omp, cuda, hip or dpcpp.'),
    ('method', 'str', 'gmres', False,
     'Krylov solver: gmres, cg, bicgstab, cgs, fcg or ir.'),
    ('precond', 'str', 'jacobi', False,
     'Preconditioner: none, jacobi or ilu.'),
    ('i_max', 'int', 1000, False,
     'The maximum number of iterations.'),
    ('eps_r', 'float', 1e-7, False,
     'The relative residual reduction factor.'),
    ('solver_args', 'dict', {}, {},
     'Raw Ginkgo solver configuration (JSON-serializable dict). If given it '
     'overrides method/precond/i_max/eps_r entirely.'),
    ('verbose', 'bool', False, False,
     'If True, print convergence information.'),
  ]

  # manapy method name -> Ginkgo solver type
  _METHODS = {
    'gmres': 'solver::Gmres',
    'cg': 'solver::Cg',
    'bicgstab': 'solver::Bicgstab',
    'cgs': 'solver::Cgs',
    'fcg': 'solver::Fcg',
    'ir': 'solver::Ir',
  }

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Intracomm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = True,
               reuse_ij: bool = True,
               with_mtx: bool = False,
               device: str = "cuda",
               method: str = "gmres",
               precond: str = "jacobi",
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

    self.pg = pg
    self.pGB = pGB

    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme,
                          reordering=reordering, solver_name=LinearSolver.SolverGinkgo)

    self._dim = self.domain.dim
    self.var = var
    self.domain.solver = "ginkgo"
    self.reordering = reordering
    self.reuse_mtx = reuse_mtx
    self.reuse_ij = reuse_ij
    self.with_mtx = with_mtx
    self.device = device
    self.method = method
    self.precond = precond
    self.i_max = i_max
    self.eps_r = eps_r
    self.solver_args = solver_args if solver_args else {}
    self.verbose = verbose

    # Ginkgo value type string and matching numpy dtype
    self._value = "double" if FLOAT_TYPE == "float64" else "float"
    self._np_value = np.float64 if FLOAT_TYPE == "float64" else np.float32
    self._Csr = getattr(pGB.matrix, f"Csr_{self._value}_int32")
    self._dense = getattr(pGB.matrix, f"dense_{self._value}")

    # Executor (only meaningful on rank 0, where the solve happens)
    self.executor = pg.device(device)

    # Ginkgo objects (built lazily on rank 0)
    self.A = None
    self.solver = None
    self.b = None
    self.x = None
    self.sol = None

    # reuse_ij: cached gather metadata / global structure
    self._ij_already_set = False
    self._gather_counts = None
    self._mat_struct = None      # (indptr, indices) of the assembled CSR

    self.rhs0 = np.zeros(self.globalsize, dtype=FLOAT_TYPE)
    # reduced rhs (BC contribution); survives across calls when presolve is skipped
    self.rhs00 = None

  # ------------------------------------------------------------------ helpers
  def _build_solver_args(self):
    """Return the Ginkgo solver configuration dict."""
    if self.solver_args:
      return self.solver_args

    if self.method not in self._METHODS:
      raise ValueError(
        f"Unknown Ginkgo method '{self.method}'. "
        f"Choices: {', '.join(self._METHODS)}")

    args = {
      "type": self._METHODS[self.method],
      "criteria": [
        {"type": "Iteration", "max_iters": self.i_max},
        {"type": "ResidualNorm", "reduction_factor": self.eps_r},
      ],
    }

    if self.precond == "jacobi":
      args["preconditioner"] = {"type": "preconditioner::Jacobi"}
    elif self.precond == "ilu":
      args["preconditioner"] = {
        "type": "preconditioner::Ilu",
        "reverse_apply": False,
        "factorization": {"type": "factorization::ParIlu"},
      }
    elif self.precond not in ("none", None):
      raise ValueError(
        f"Unknown Ginkgo preconditioner '{self.precond}'. "
        "Choices: none, jacobi, ilu.")

    return args

  def _gather_to_root(self):
    """Gather the distributed (row, col, data) triplets onto rank 0.

    Returns (row, col, data) on rank 0 (global 0-based indices) and
    (None, None, None) on the other ranks.
    """
    comm = self.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_row = np.ascontiguousarray(self._row, dtype=np.int32)
    local_col = np.ascontiguousarray(self._col, dtype=np.int32)
    local_data = np.ascontiguousarray(self._data, dtype=self._np_value)

    if size == 1:
      return local_row, local_col, local_data

    if not (self.reuse_ij and self._gather_counts is not None):
      n_local = np.array([local_row.size], dtype=np.int32)
      counts = np.empty(size, dtype=np.int32) if rank == 0 else None
      comm.Gather(n_local, counts, root=0)
      self._gather_counts = counts

    counts = self._gather_counts
    if rank == 0:
      total = int(np.sum(counts))
      row = np.empty(total, dtype=np.int32)
      col = np.empty(total, dtype=np.int32)
      data = np.empty(total, dtype=self._np_value)
    else:
      row = col = data = None

    mpi_val = MPI.DOUBLE if self._np_value == np.float64 else MPI.FLOAT
    comm.Gatherv(local_row, (row, counts), root=0)
    comm.Gatherv(local_col, (col, counts), root=0)
    comm.Gatherv([local_data, mpi_val], (data, counts) if rank == 0 else None, root=0)

    return row, col, data

  # ------------------------------------------------------------------ solve
  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]] = None):

    self.presolve(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx,
                  reuse_ij=self.reuse_ij)

    if rhs is not None:
      rhs = rhs + self.rhs00 if self.rhs00 is not None else rhs
    else:
      rhs = self.rhs00

    if self.comm.Get_rank() == 0:
      # Right-hand side (warm-start the solution with the previous iterate)
      b_np = np.ascontiguousarray(rhs, dtype=self._np_value).reshape(self.globalsize, 1)
      self.b = self._dense(self.executor, b_np)
      if self.x is None:
        self.x = self._dense(self.executor, (self.globalsize, 1))
        self.x.fill(0.0)

      solver_args = self._build_solver_args()

      if self.reuse_mtx and self.solver is not None:
        # Reusable solver bound to a constant matrix: just apply it.
        self.solver.apply(self.b, self.x)
        result = self.x
        logger = None
      else:
        logger, result = self.pg.solve(self.A, self.b, self.x,
                                       solver_args=solver_args)
        # keep the produced solution for the next warm start
        self.x = result

      self.sol = np.array(result.copy_to_host()).reshape(-1).astype(FLOAT_TYPE)

      if self.verbose and logger is not None:
        print(f"[Ginkgo] {self.method}/{self.precond} converged="
              f"{logger.has_converged()} iters={logger.get_num_iterations()}")

      if self.reordering and self.comm.Get_size() == 1:
        self.sol = self.sol[np.argsort(self.perm)]

      # Convert solution for scattering
      self.convert_solution(self.sol, self.x1converted, self.domain.cells.tc,
                            self.globalsize)

    self.comm.Scatterv(sendbuf=[self.x1converted, self.sendcounts1, self.mpi_precision],
                       recvbuf=self.var.cell, root=0)

  def presolve(self, reuse_mtx=False, with_mtx=False, reuse_ij=False):
    if reuse_mtx and self.A is not None:
      # Matrix (and solver) already built; nothing to reassemble.
      return

    self.update_ghost_values()

    # assembly row, col, data, rhs(bc)
    if not with_mtx:
      self.assembly()
    elif not np.any(self._data):
      raise ValueError(
        "with_mtx=True but no matrix was provided: the solver expects you to "
        "fill the (row, col, data) triplets yourself. Call "
        "set_matrix(row, col, data) (or fill _row/_col/_data directly) before "
        "solving, or use with_mtx=False to assemble the matrix automatically.")

    if self.reordering and self.comm.Get_size() == 1:
      print("=>Reordering the matrix ...")
      self.reordering_matrix()

    # Global RHS contribution from the boundary conditions, summed on root.
    self.rhs00 = self.comm.reduce(self.rhs0, op=MPI.SUM, root=0)

    # Gather the global matrix on rank 0 and build the Ginkgo operator there.
    row, col, data = self._gather_to_root()

    if self.comm.Get_rank() == 0:
      # csr_matrix sums duplicate (i, j) entries, which is required since the
      # assembly produces several contributions per coefficient.
      mat = csr_matrix((data, (row, col)),
                       shape=(self.globalsize, self.globalsize))
      mat.sort_indices()
      self._mat_struct = (mat.indptr, mat.indices)
      self.A = self._Csr(self.executor, mat)

      if self.reuse_mtx:
        # Pre-build a reusable solver bound to this (constant) matrix.
        self.solver = self.pg.generate_solver(self.A,
                                              solver_args=self._build_solver_args())

    self._ij_already_set = True

  def view(self):
    """Print basic information about the assembled Ginkgo system (rank 0)."""
    if self.comm.Get_rank() == 0 and self.A is not None:
      print(f"[Ginkgo] device={self.device} value={self._value} "
            f"size={self.globalsize} nnz={self.A.get_num_stored_elements()} "
            f"method={self.method} precond={self.precond}")

  def set_matrix(self, row, col, data):
    """Provide the matrix triplets explicitly (use together with with_mtx=True).

    The (row, col) indices must be 0-based global indices.
    """
    self._row = np.asarray(row, dtype=np.int32)
    self._col = np.asarray(col, dtype=np.int32)
    self._data = np.asarray(data, dtype=FLOAT_TYPE)

  def clear(self):
    self.A = None
    self.solver = None
    self.b = None
    self.x = None
    self.sol = None
    self._ij_already_set = False
    self._gather_counts = None
    self._mat_struct = None
