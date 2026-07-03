import numpy as np
import numpy.typing as npt
from typing import Union
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import scipy.sparse.linalg.dsolve as sls
import sys
from scipy import sparse

class ScipySolver(LinearSolver):
  _parameters = [
    ('scheme', 'str', 'diamond', 'diamond',
     'scheme: diamond, fv (orthogonal) or fv_corrected (non-orthogonal corrected).'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, pre-factorize the matrix.'),
    ('memory_relaxation', 'int', 20, False,
     'The percentage increase in the estimated working space.'),
    ('verbose', 'bool', False, False,
     """If True, the solver can print more information about the                                                                                                                                       
        solution.""")
  ]

  def __init__(self,
               domain: Domain,
               var: Variable,
               method: str = "superlu",
               comm:MPI.Intracomm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               reuse_mtx: bool = True
               ):

    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme, reordering=reordering, solver_name=LinearSolver.SolverScipy)
    if self.comm.Get_size() > 1:
      raise ValueError('ScipySolver is not parallel! use MUMPSSolver or PETScKrylovSolver')
    self.sls = sls
    self.reuse_mtx = reuse_mtx
    self.reordering = reordering
    self.solve = None
    self.mat = None

    if method in ['auto', 'umfpack']:
      try:
        import scikits.umfpack as umfpack
        assert hasattr(umfpack, 'UMFPACK_OK')
      except ImportError:
        print('cannot import scikits.umfpack direct solvers!')
        sys.exit(1)
      is_umfpack = True
    elif method == 'superlu':
      is_umfpack = False
    else:
      raise ValueError(f'uknown solution method! ({method})')

    if is_umfpack:
      self.sls.use_solver(useUmfpack=True, assumeSortedIndices=True)
    else:
      self.sls.use_solver(useUmfpack=False)

    self.clear()

    self._domain = domain
    self._dim = self._domain.dim
    self.var = var
    self._domain.solver = "scipy"

    self.rhs0 = np.zeros(self.globalsize, dtype=FLOAT_TYPE)


  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]]=None):
    if not self.reuse_mtx:
      self.clear()

    self.presolve(reuse_mtx=self.reuse_mtx)

    if rhs is not None:
      rhs += self.rhs0
    else:
      rhs = self.rhs0

    if self.reuse_mtx:
      self.sol = self.solve(rhs)
    else:
      self.sol = self.sls.spsolve(self.mat, rhs)

    if self.reordering and self.comm.Get_size() == 1:
      self.var.cell = self.var.cell[np.argsort(self.perm)]

  def presolve(self, reuse_mtx=False):
    if not reuse_mtx or (self.solve is None):
      self.update_ghost_values()

      # assembly row, col , data, rhs(bc)
      self.assembly()
      if self.reordering:
        self.reordering_matrix()

      self.mat = sparse.csc_matrix((self._data, (self._row, self._col)))
      self.solve = self.sls.factorized(self.mat)

  def clear(self):
    self.solve = None
