import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import numpy.typing as npt
from typing import Union

# TODO make the solver work also with float32
# TODO fix reuse_mtx and verbose

class MUMPSSolver(LinearSolver):
  _parameters = [
    ('scheme', 'str', 'diamond', 'diamond',
     'scheme: diamond, fv (orthogonal) or fv_corrected (non-orthogonal corrected).'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, pre-factorize the matrix.'),
    ('reuse_ij', 'bool', True, False,
     'If True, set the (row, col) structure and run analysis only once, '
     'then only refresh the numerical values on subsequent calls.'),
    ('non_orthogonal_corrections', 'int', 2, False,
     'Number of explicit non-orthogonal correction solves for scheme=fv_corrected.'),
    ('non_orthogonal_limiter', 'float', 1.0, False,
     'Blend psi in [0,1] of the non-orthogonal correction (scheme=fv_corrected): 1=full, 0=none.'),
    ('with_mtx', 'bool', False, True,
     'If True, the matrix should be given.'),
    ('system', "str", "double", "double",
     'Mumps precision'),
    ('memory_relaxation', 'int', 20, False,
     'The percentage increase in the estimated working space.'),
    ('blr', 'bool', False, False,
     'If True, enable Block Low-Rank (BLR) compression.'),
    ('blr_eps', 'float', 1e-4, False,
     'BLR dropping tolerance (CNTL(7)).'),
    ('verbose', 'bool', False, False,
     """If True, the solver can print more information about the
        solution.""")
  ]

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Intracomm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = False,
               reuse_ij: bool = True,
               non_orthogonal_corrections: int = 2,
               non_orthogonal_limiter: float = 1.0,
               with_mtx: bool = False,
               system: str = "double",
               memory_relaxation: int = 20,
               blr: bool = False,
               blr_eps: float = 1e-4,
               spd: bool = False,
               ):

    import mumps4py.mumps_solver as mumps

    self.mumps = mumps
    self.mumps_ls = None

    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme, reordering=reordering, solver_name=LinearSolver.SolverMumps, non_orthogonal_corrections=non_orthogonal_corrections, non_orthogonal_limiter=non_orthogonal_limiter)


    self.system = system

    self._dim = self.domain.dim
    self.var = var
    self.domain.solver = "mumps"
    self.reordering = reordering
    self.reuse_mtx = reuse_mtx
    self.reuse_ij = reuse_ij
    self.with_mtx = with_mtx
    self.memory_relaxation = memory_relaxation
    self.blr = blr
    self.blr_eps = blr_eps
    # negate (A,b) in the shared assembly -> SPD. MUMPS here uses unsymmetric LU
    # (sym=0), which factorizes either sign fine, so spd is not REQUIRED for MUMPS;
    # it only matters if MUMPS is switched to its SPD Cholesky mode (sym=1).
    self.spd = spd
    self.sol = None

    # State for reuse_ij: structure set / analysis done only once
    self._ij_already_set = False
    self._analysis_done = False
    self._row_origin = None
    self._col_origin = None

    self.rhs0 = np.zeros(self.globalsize, dtype=FLOAT_TYPE)
    # reduced rhs (BC contribution); kept as an attribute so it survives across
    # calls when presolve is skipped (reuse_mtx=True in a time loop)
    self.rhs00 = None

  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]]=None):

    # clear() is no longer called automatically: with reuse_ij the MUMPS
    # instance (structure + analysis) must persist across calls even when
    # reuse_mtx is False (re-factorization each call). Call clear() manually
    # for a full reset.

    self.presolve(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx,
                  reuse_ij=self.reuse_ij)

    user_rhs = rhs
    if user_rhs is not None:
      # user_rhs is LOCAL (length localsize, indexed by local cell), like the PETSc
      # backend expects. MUMPS uses a centralised GLOBAL rhs, so scatter it to the
      # global positions (cells.loctoglob) and reduce to root -- consistent with
      # rhs00 (the reduced global BC source). In serial loctoglob is the identity so
      # this reduces to the former `user_rhs + rhs00`.
      g = np.zeros(self.globalsize, dtype=FLOAT_TYPE)
      g[self.domain.cells.loctoglob] = user_rhs
      g = self.comm.reduce(g, op=MPI.SUM, root=0)
      solve_rhs = (g + self.rhs00) if self.comm.Get_rank() == 0 else None
    else:
      solve_rhs = self.rhs00

    ncycles = 1 + self.non_orthogonal_corrections
    for cycle in range(ncycles):
      if cycle > 0:
        solve_rhs = self.fv_corrected_rhs(self.rhs00, user_rhs, global_rhs=True)

      # Allocation size of rhs
      if self.comm.Get_rank() == 0:
        self.sol = solve_rhs.copy()
        self.mumps_ls.set_rhs_centralized(self.sol)

      # Solution Phase
      self.mumps_ls._mumps_call(job=3)

      if self.reordering and self.comm.Get_size() == 1:
        self.sol = self.sol[np.argsort(self.perm)]

      if self.comm.Get_rank() == 0:
        # Convert solution for scattering
        self.convert_solution(self.sol, self.x1converted, self.domain.cells.tc, self.globalsize)

      self.comm.Scatterv(sendbuf=[self.x1converted, self.sendcounts1, self.mpi_precision], recvbuf=self.var.cell,
                         root=0)

      if self.scheme not in self.fv_schemes:
        break

  def presolve(self, reuse_mtx=False, with_mtx=False, reuse_ij=False):
    if not reuse_mtx or self.mumps_ls is None:
      self.update_ghost_values()

      # assembly row, col , data, rhs(bc)
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

      # SPD path: MUMPS sym=1 (Cholesky) expects ONLY the lower triangle (row>=col);
      # manapy assembles the FULL symmetric matrix, so keep row>=col to avoid summing
      # the (i,j)+(j,i) duplicates (which would double the off-diagonals). Deterministic
      # (same mask every call) -> compatible with reuse_ij. The matrix is already
      # negated to SPD by the shared assembly (spd flag). 0-based indices here.
      if self.spd:
        mask = self._row >= self._col
        self._row = np.ascontiguousarray(self._row[mask])
        self._col = np.ascontiguousarray(self._col[mask])
        self._data = np.ascontiguousarray(self._data[mask])

      self.rhs00 = self.comm.reduce(self.rhs0, op=MPI.SUM, root=0)

      if reuse_ij:
        if not self._ij_already_set:
          if self.mumps_ls is None:
            self.mumps_ls = self.mumps.MumpsSolver(verbose=self.verbose, system=self.system,
                                                   sym=(1 if self.spd else 0),
                                                   mem_relax=self.memory_relaxation)
          # Fortran indexing + keep a copy. Register the PERSISTENT copies with
          # MUMPS: mumps4py stores a RAW ctypes pointer with no python reference,
          # so the registered arrays must outlive every later set_matrix()/assembly()
          # (registering self._row dangles as soon as set_matrix replaces it -- the
          # GC frees the buffer and MUMPS reads recycled memory -> error -53).
          self._row += 1
          self._col += 1
          self._row_origin = self._row.copy()
          self._col_origin = self._col.copy()

          self.mumps_ls.set_rc_distributed(self._row_origin, self._col_origin,
                                           self.globalsize)
          self._ij_already_set = True
        else:
          # structure registered once; only the values may change. The provided /
          # re-assembled triplets MUST come in the registered order (guaranteed by
          # the deterministic assembly loops); verify the pattern cheaply.
          if len(self._row) != len(self._row_origin):
            raise ValueError(
              f"reuse_ij=True but the matrix pattern changed: {len(self._row)} "
              f"entries vs {len(self._row_origin)} registered. Call clear() first "
              "or use reuse_ij=False for matrices with a changing structure.")
          if with_mtx and (not np.array_equal(self._row + 1, self._row_origin)
                           or not np.array_equal(self._col + 1, self._col_origin)):
            raise ValueError(
              "reuse_ij=True but set_matrix() provided a different sparsity "
              "pattern (or a different entry order) than the registered one. "
              "Call clear() first or use reuse_ij=False.")
          if not with_mtx:
            # assembly() rewrote _row/_col as 0-based (same buffers, same order)
            self._row[:] = self._row_origin
            self._col[:] = self._col_origin
      else:
        if self.mumps_ls is None:
          self.mumps_ls = self.mumps.MumpsSolver(verbose=self.verbose, system=self.system,
                                                 sym=(1 if self.spd else 0),
                                                 mem_relax=self.memory_relaxation)
        # Fortran indexing
        self._row += 1
        self._col += 1
        self.mumps_ls.set_rc_distributed(self._row, self._col, self.globalsize)

      self.mumps_ls.set_data_distributed(self._data, self.globalsize)

      self.mumps_ls.set_icntl(18, 3)
      self.mumps_ls.set_icntl(16, 1)
      self.mumps_ls.set_icntl(7, 7)      # ordering (METIS/SCOTCH if available)
      self.mumps_ls.set_icntl(22, 0)     # out-of-core off (default)
      if self.blr:
        self.mumps_ls.set_icntl(35, 1)            # activate BLR
        self.mumps_ls.set_cntl(7, self.blr_eps)   # BLR dropping tolerance

      if self.comm.Get_rank() == 0:
        self.sol = self.rhs00.copy()

      # Analyse (only once when reuse_ij)
      if not self._analysis_done:
        self.mumps_ls._mumps_call(job=1)
        if reuse_ij:
          self._analysis_done = True

      # Factorization Phase
      self.mumps_ls._mumps_call(job=2)

      # Allocation size of rhs
      if self.comm.Get_rank() == 0:
        self.mumps_ls.set_rhs_centralized(self.sol)
      else:
        self.sol = np.zeros(self.globalsize, dtype=FLOAT_TYPE)

  def set_matrix(self, row, col, data):
    """Provide the matrix triplets explicitly (use together with with_mtx=True).

    The (row, col) indices must be 0-based; Fortran indexing is handled
    internally. Call this before solving.
    """
    self._row = np.asarray(row, dtype=np.int32)
    self._col = np.asarray(col, dtype=np.int32)
    self._data = np.asarray(data, dtype=FLOAT_TYPE)

  def clear(self):
    self.mumps_ls = None
    self._ij_already_set = False
    self._analysis_done = False
    self._row_origin = None
    self._col_origin = None
    
  def view(self):
      self.view_matrix_on_root()

  def view_matrix_on_root(self, threshold=1e-10):
    """
    Gather the distributed sparse matrix (row, col, data) from all processes
    onto rank 0 and rebuild it as a CSR matrix for display/inspection.
    """
    local_row = self._row.astype(np.int32)
    local_col = self._col.astype(np.int32)
    local_data = self._data.astype(np.float64)

    comm = self.comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: gather local sizes
    n_local = np.array([len(local_row)], dtype=np.int32)
    n_sizes = None
    if rank == 0:
      n_sizes = np.empty(size, dtype=np.int32)
    comm.Gather(n_local, n_sizes, root=0)

    # Step 2: gather the data (row, col, data)
    row_recv = None
    col_recv = None
    data_recv = None

    if rank == 0:
      total_nnz = np.sum(n_sizes)
      row_recv = np.empty(total_nnz, dtype=np.int32)
      col_recv = np.empty(total_nnz, dtype=np.int32)
      data_recv = np.empty(total_nnz, dtype=np.float64)

    comm.Gatherv(local_row, (row_recv, n_sizes), root=0)
    comm.Gatherv(local_col, (col_recv, n_sizes), root=0)
    comm.Gatherv(local_data, (data_recv, n_sizes), root=0)

    # Step 3: build the matrix on rank 0
    if rank == 0:
      from scipy.sparse import coo_matrix
      from scipy.sparse import find
      # _row/_col are 1-based (Fortran) once the structure has been set
      mat = coo_matrix((data_recv, (row_recv - 1, col_recv - 1)),
                       shape=(self.globalsize, self.globalsize))
      csr = mat.tocsr()
      print("Global matrix (CSR format):")
      row, col, data = find(csr)
      for i, j, v in zip(row, col, data):
        if abs(v) > threshold:
          print(f"A[{i}, {j}] = {v:.14f}")
      return csr
    else:
      return None
