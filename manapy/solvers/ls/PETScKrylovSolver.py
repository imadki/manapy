import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import numpy.typing as npt
from typing import Union

class PETScKrylovSolver(LinearSolver):
  _parameters = [
    ('scheme', 'str', 'diamond', 'diamond',
     'scheme: diamond, fv (orthogonal) or fv_corrected (non-orthogonal corrected).'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, pre-factorize the matrix.'),
    ('reuse_ij', 'bool', True, False,
     'If True, reuse the matrix structure (preallocation) and only refresh '
     'the values with zeroEntries on subsequent calls.'),
    ('non_orthogonal_corrections', 'int', 2, False,
     'Number of explicit non-orthogonal correction solves for scheme=fv_corrected.'),
    ('non_orthogonal_limiter', 'float', 1.0, False,
     'Blend psi in [0,1] of the non-orthogonal correction (scheme=fv_corrected): 1=full, 0=none.'),
    ('with_mtx', 'bool', False, True,
     'If True, the matrix should be given.'),
    ('method', 'str', 'fgmres', False,
     'The actual ksp solver to use.'),
    ('precond', 'str', 'gamg', False,
     'The preconditioner.'),
    ('sub_precond', 'str', 'none', False,
     'The preconditioner for matrix blocks (in parallel runs).'),
    ('factor_solver', 'str', 'none', False,
     'Use Factor solver type such as mumps, superlu_dist'),
    ('i_max', 'int', 1000, False,
     'The maximum number of iterations.'),
    ('eps_a', 'float', 1e-6, False,
     'The absolute tolerance for the residual.'),
    ('eps_r', 'float', 1e-12, False,
     'The relative tolerance for the residual.'),
    ('eps_d', 'float', 1e5, False,
     'The divergence tolerance for the residual.'),
    ('petsc_options', 'dict', {}, {},
     """Additional parameters supported by the method. Can be used to pass
        all PETSc options supported by :func:`petsc.Options()`."""),
    ('spd', 'bool', False, False,
     'If True, negate the assembled system (A,b) -> (-A,-b). manapy assembles the '
     'Laplacian in its natural div(grad) sign (negative diagonal = symmetric '
     'NEGATIVE definite); Cholesky-based preconditioners (icc) need a POSITIVE '
     'definite matrix. Negating gives the same solution but a SPD matrix so icc '
     'works. Only valid for symmetric schemes (fv/orthogonal); leave False for '
     'diamond (non-symmetric).'),
    ('mat_type', 'str', 'mpiaij', False,
     'PETSc matrix type. "mpiaij" (default) stores both triangles. "mpisbaij" '
     'stores only the upper triangle of a SYMMETRIC matrix -> half the memory and '
     '~1.5x faster SpMV in CG. Only valid for symmetric schemes (fv); do NOT use '
     'with the non-symmetric diamond scheme.'),

  ]

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Comm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = True,
               reuse_ij: bool = True,
               non_orthogonal_corrections: int = 2,
               non_orthogonal_limiter: float = 1.0,
               with_mtx: bool = False,
               method: str = 'fgmres',
               precond: str = 'gamg',
               sub_precond: str = 'none',
               factor_solver: str = 'none',
               i_max: int = 1000,
               eps_a: float = 1e-6,
               eps_r: float = 1e-12,
               eps_d: float = 1e5,
               spd: bool = False,
               mat_type: str = "mpiaij",
               petsc_options: dict = None
   ):
    if petsc_options is None:
      # Prevent Problem: mutable default argument
      petsc_options = {}

    try:
      import petsc4py
    except ImportError:
      import sys
      print("petsc4py is not installed. Please install it first.")
      sys.exit(1)

    from petsc4py import PETSc
    import sys
    petsc4py.init(sys.argv)

    self.petsc = PETSc
    self.ksp = None
    self.reuse_mtx = reuse_mtx
    self.reuse_ij = reuse_ij
    self._ij_already_set = False
    self.NNZ = None
    self.with_mtx = with_mtx
    self.sub_precond = sub_precond
    self.reordering = reordering
    self.method = method
    self.factor_solver = factor_solver
    self.precond = precond
    self.eps_a = eps_a
    self.eps_r = eps_r
    self.eps_d = eps_d
    self.spd = spd
    self.mat_type = mat_type
    self.i_max = i_max
    self.petsc_options = petsc_options
    self.sol = None
    self.sendcounts2 = None
    self.recvbuf = None
    self.rhs = None
    self.mat = None

    self.converged_reasons = {}
    for key, val in PETSc.KSP.ConvergedReason.__dict__.items():
      if isinstance(val, int):
        self.converged_reasons[val] = key


    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme, reordering=reordering, solver_name=LinearSolver.SolverPetsc, non_orthogonal_corrections=non_orthogonal_corrections, non_orthogonal_limiter=non_orthogonal_limiter)

    self._domain = domain
    self._dim = self._domain.dim
    self.var = var
    self._domain.solver = "petsc"
    self.rhs0 = np.zeros(self.localsize, dtype=FLOAT_TYPE)
    # cached global row indices (int32) for the vectorised RHS assembly.
    self._l2g_i32 = np.asarray(self.domain.cells.loctoglob, dtype=np.int32)
    # vectorised matrix value assembly via a global-row CSR + setValuesCSR (one
    # C-level call). Set to False to fall back to the per-entry setValues loop.
    self._mat_csr = True



  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]]=None):

    def custom_monitor(ksp, its, r_norm):
      print(f"Iteration {its}: Residual Norm = {r_norm}")

    if not self.reuse_mtx and not self.reuse_ij:
      self.clear()

    self.create_petsc_matrix(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx,
                             reuse_ij=self.reuse_ij)

    user_rhs = rhs

    # For scheme=fv_corrected the non-orthogonal correction is an explicit (deferred) term
    # rebuilt from the current solution gradient; cycle 0 solves the orthogonal
    # part, each extra cycle re-solves the SAME matrix with the corrected RHS.
    ncycles = 1 + self.non_orthogonal_corrections
    for cycle in range(ncycles):
      if cycle == 0:
        # Local RHS = BC source (rhs0) [+ user rhs]; set explicitly so a reused
        # matrix (which skips create_rhs) still starts from a fresh RHS.
        if user_rhs is not None:
          self.update_rhs(rhs=user_rhs)
        else:
          self.create_rhs()
      else:
        # rhs0 + non-orthogonal correction (+ user rhs), all local (loc kernel).
        corr = self.fv_correction_rhs(global_rhs=False)
        extra = corr if user_rhs is None else corr + user_rhs
        self.update_rhs(rhs=extra)

      # warm-start: reuse the previous solution as initial guess
      self.initiate_sol()
      self.ksp.solve(self.rhs, self.sol)

      if self.verbose:
        print(self.ksp.getType(), self.ksp.getPC().getType(), self.sub_precond,
              self.ksp.reason, self.converged_reasons[self.ksp.reason],
              self.ksp.getIterationNumber())

        for i in range(self.ksp.getIterationNumber()):
          r_norm = self.ksp.getResidualNorm()
          custom_monitor(self.ksp, i + 1, r_norm)

      if self.reordering and self.comm.Get_size() == 1:
        self.sol.array = self.sol.array[np.argsort(self.perm)]

      self.comm.Gatherv(sendbuf=self.sol.array.astype(FLOAT_TYPE), recvbuf=(self.recvbuf, self.sendcounts2),
                        root=0)

      if self.comm.Get_rank() == 0:
        # Convert solution for scattering
        ls_compute.convert_solution(self.recvbuf, self.x1converted, self.domain.cells.tc, self.globalsize)

      # Scatter back to var.cell so the next cycle's correction sees this iterate.
      self.comm.Scatterv([self.x1converted, self.sendcounts1, self.mpi_precision], self.var.cell, root=0)

      if self.scheme not in self.fv_schemes:
        break

  def create_ksp(self, options: dict, comm: MPI.Comm):
    optDB = self.petsc.Options()

    # MPI compatibility with CUDA
    optDB['use_gpu_aware_mpi'] = 0

    optDB['sub_pc_type'] = self.sub_precond

    for key, val in options.items():
      optDB[key] = val

    self.ksp = self.petsc.KSP()
    self.ksp.create(comm)

    self.ksp.setType(self.method)
    pc = self.ksp.getPC()

    if self.precond == "lu" and self.factor_solver == 'none':
      self.factor_solver = "mumps"
      optDB['pc_factor_mat_ordering_type'] = "rcm"

    pc.setType(self.precond)
    if self.factor_solver is not None:
      pc.setFactorSolverType(self.factor_solver)

    self.ksp.setFromOptions()

  def update_rhs(self, rhs=None):
    # vectorised: one setValues over the global-index array instead of a per-cell
    # Python loop (loctoglob routes off-process entries via assembly, as before).
    self.rhs = self.mat.getVecLeft()
    self.rhs.setValues(self._l2g_i32, self.rhs0 + rhs)
    self.rhs.assemblyBegin()
    self.rhs.assemblyEnd()

  def create_rhs(self):
    self.rhs = self.mat.getVecLeft()
    self.rhs.setValues(self._l2g_i32, self.rhs0)
    self.rhs.assemblyBegin()
    self.rhs.assemblyEnd()

  def initiate_sol(self):
    self.ksp.setInitialGuessNonzero(True)

  def create_petsc_matrix(self, reuse_mtx=False, with_mtx=False, reuse_ij=False):
    if not reuse_mtx or (self.ksp is None):
      ###################################################################
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

      ###################################################################
      # reordering matrix
      if self.reordering and self.comm.Get_size() == 1:
        self.reordering_matrix()

      ###################################################################
      # Create the petsc matrix structure (preallocation) only once when
      # reuse_ij is enabled; afterwards only the values are refreshed.
      if not reuse_ij or not self._ij_already_set:
        # non-zeros per row = number of UNIQUE (row,col) pairs, NOT the raw triplet
        # count. The diamond assembly emits many duplicate (row,col) triplets (summed
        # later by ADD_VALUES), so np.unique(_row) alone over-counts wildly (max ~3245
        # on a 2M-cell triangle mesh vs a true ~30). setPreallocationNNZ(max(NNZ)) then
        # asks PETSc to allocate max*n_local_rows entries, whose product overflows
        # 32-bit PetscInt at p>=2 (3245*1e6 = 3.2e9 > INT32_MAX). Dedup (row,col) first
        # (encode the pair as row*globalsize+col so a single 1-D np.unique dedups it).
        key = (np.asarray(self._row, dtype=np.int64) * np.int64(self.globalsize)
               + np.asarray(self._col, dtype=np.int64))
        rows_u = (np.unique(key) // np.int64(self.globalsize)).astype(np.int32)
        unique, counts = np.unique(rows_u, return_counts=True)

        NNZ_loc = np.zeros(self.globalsize, dtype=np.int32)
        for uid, count in zip(unique, counts):
          NNZ_loc[uid] = count

        self.NNZ = np.zeros(self.globalsize, dtype=np.int32)
        self.comm.Allreduce(NNZ_loc, self.NNZ, op=MPI.SUM)  # , root=0)

        self.mat = self.petsc.Mat().create()
        self.mat.setSizes(self.globalsize)
        self.mat.setType(self.mat_type)
        # SBAIJ stores only the UPPER triangle of a symmetric matrix -> half the
        # memory and ~1.5x faster (memory-bound) SpMV in CG. Block size 1; the full
        # triplets are still fed via setValues, IGNORE_LOWER_TRIANGULAR drops the
        # lower half. Only valid for SYMMETRIC matrices (fv + spd); the caller must
        # not select sbaij for the non-symmetric diamond scheme. ORDER MATTERS:
        # setBlockSize BEFORE setFromOptions, IGNORE_LOWER_TRIANGULAR AFTER
        # preallocation (setting it before preallocation segfaults PETSc).
        if "sbaij" in self.mat_type:
          self.mat.setBlockSize(1)
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ(max(self.NNZ))
        if "sbaij" in self.mat_type:
          self.mat.setOption(self.petsc.Mat.Option.IGNORE_LOWER_TRIANGULAR, True)
        self.mat.setOption(self.petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self._ij_already_set = True

      ###################################################################
      # Refresh the numerical values. The triplet (row, col) indices are GLOBAL
      # (loctoglob / halosext) and manapy does NOT number cells contiguously per
      # rank, so PETSc's bulk CSR/COO interfaces (which want the owned rows in
      # [rstart, rend)) don't apply -- PETSc keeps its own ownership split and
      # routes off-process entries via the stash. We still avoid the per-non-zero
      # Python loop by grouping the triplets by row and setting each row in ONE
      # setValues call (one call per owned row; duplicate columns are summed by
      # ADD_VALUES within the call). `_mat_csr=False` restores the per-entry loop.
      self.mat.zeroEntries()
      ADD = self.petsc.InsertMode.ADD_VALUES
      if self._mat_csr:
        row = np.asarray(self._row, dtype=np.int32)
        col = np.asarray(self._col, dtype=np.int32)
        data = np.asarray(self._data, dtype=FLOAT_TYPE)
        order = np.argsort(row, kind="stable")
        row = row[order]; col = col[order]; data = data[order]
        # row-change boundaries -> [start, end) slices of one global row each
        cut = np.concatenate(([0], np.nonzero(np.diff(row))[0] + 1, [row.size]))
        for k in range(cut.size - 1):
          s = cut[k]; e = cut[k + 1]
          self.mat.setValues(row[s], col[s:e], data[s:e], addv=ADD)
      else:
        for i in range(len(self._row)):
          self.mat.setValues(self._row[i], self._col[i], self._data[i], addv=True)
      self.mat.assemblyBegin(self.mat.AssemblyType.FINAL)
      self.mat.assemblyEnd(self.mat.AssemblyType.FINAL)

      ###################################################################
      if self.sol is None:
        self.sol = self.mat.getVecRight()
        self.sendcounts2 = np.array(self.comm.gather(len(self.sol.array), root=0))

      if self.comm.Get_rank() == 0:
        self.recvbuf = np.empty(sum(self.sendcounts2), dtype=FLOAT_TYPE)
      else:
        self.recvbuf = None
      ###################################################################
      # Create the ksp solver
      ###################################################################

      self.create_ksp(options=self.petsc_options, comm=self.comm)
      self.ksp.setOperators(self.mat)
      self.ksp.setTolerances(atol=self.eps_a, rtol=self.eps_r, divtol=self.eps_d,
                             max_it=self.i_max)
      self.ksp.setFromOptions()
      ###################################################################

      #            #Create the solution
      #            self.Initiate_sol()

      # Create the Rhs
      self.create_rhs()

  def view(self):
    self.mat.view()
    self.rhs.view()
    self.sol.view()

  def set_matrix(self, row, col, data):
    """Provide the matrix triplets explicitly (use together with with_mtx=True).

    The (row, col) indices must be 0-based global indices. Call this before
    solving.
    """
    self._row = np.asarray(row, dtype=np.int32)
    self._col = np.asarray(col, dtype=np.int32)
    self._data = np.asarray(data, dtype=FLOAT_TYPE)

  def clear(self):
    self.ksp = None
    self._ij_already_set = False
    self.mat = None
    self.sol = None
    self.sendcounts2 = None
