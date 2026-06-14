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
    ('scheme', 'str', 'diamond', 'fv4',
     'scheme diamond or fv4.'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, pre-factorize the matrix.'),
    ('reuse_ij', 'bool', True, False,
     'If True, reuse the matrix structure (preallocation) and only refresh '
     'the values with zeroEntries on subsequent calls.'),
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

  ]

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Comm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = True,
               reuse_ij: bool = True,
               with_mtx: bool = False,
               method: str = 'fgmres',
               precond: str = 'gamg',
               sub_precond: str = 'none',
               factor_solver: str = 'none',
               i_max: int = 1000,
               eps_a: float = 1e-6,
               eps_r: float = 1e-12,
               eps_d: float = 1e5,
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


    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme, reordering=reordering, solver_name=LinearSolver.SolverPetsc)

    self._domain = domain
    self._dim = self._domain.dim
    self.var = var
    self._domain.solver = "petsc"
    self.rhs0 = np.zeros(self.localsize, dtype=FLOAT_TYPE)



  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]]=None):

    def custom_monitor(ksp, its, r_norm):
      print(f"Iteration {its}: Residual Norm = {r_norm}")

    if not self.reuse_mtx and not self.reuse_ij:
      self.clear()

    self.create_petsc_matrix(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx,
                             reuse_ij=self.reuse_ij)

    if rhs is not None:
      self.update_rhs(rhs=rhs)

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

    self.comm.Scatterv([self.x1converted, self.sendcounts1, self.mpi_precision], self.var.cell, root=0)

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
    self.rhs = self.mat.getVecLeft()
    for i in range(self.domain.nbcells):
      self.rhs.setValues(self.domain.cells.loctoglob[i], self.rhs0[i] + rhs[i])

    self.rhs.assemblyBegin()
    self.rhs.assemblyEnd()

  def create_rhs(self):

    self.rhs = self.mat.getVecLeft()
    for i in range(self.domain.nbcells):
      self.rhs.setValues(self.domain.cells.loctoglob[i], self.rhs0[i])

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
        # non zero values for each rows
        NNZ_loc = np.zeros(self.globalsize, dtype=np.int32)
        unique, counts = np.unique(np.asarray(self._row, dtype=np.int32),
                                   return_counts=True)

        for uid, count in zip(unique, counts):
          NNZ_loc[uid] = count

        self.NNZ = np.zeros(self.globalsize, dtype=np.int32)
        self.comm.Allreduce(NNZ_loc, self.NNZ, op=MPI.SUM)  # , root=0)

        self.mat = self.petsc.Mat().create()
        self.mat.setSizes(self.globalsize)
        self.mat.setType("mpiaij")
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ(max(self.NNZ))
        self.mat.setOption(self.petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self._ij_already_set = True

      ###################################################################
      # Refresh the numerical values
      self.mat.zeroEntries()
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
