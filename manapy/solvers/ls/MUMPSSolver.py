import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.backends.types import FLOAT_TYPE
from manapy.solvers.ls.LinearSolver import LinearSolver
import numpy.typing as npt
from typing import Union

# TODO make the solver work also with float32
# TODO fix reuse_mtx and verbose

class MUMPSSolver(LinearSolver):
  _parameters = [
    ('scheme', 'str', 'diamond', 'fv4',
     'scheme diamond or fv4.'),
    ('reordering', 'bool', False, False,
     'reordering the matrix only in serial case.'),
    ('reuse_mtx', 'bool', True, False,
     'If True, pre-factorize the matrix.'),
    ('with_mtx', 'bool', False, True,
     'If True, the matrix should be given.'),
    ('system', "str", "double", "double",
     'Mumps precision'),
    ('memory_relaxation', 'int', 20, False,
     'The percentage increase in the estimated working space.'),
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
               with_mtx: bool = True,
               system: str = "double",
               memory_relaxation: int = 20,
               ):

    import mumps4py.mumps_solver as mumps

    self.mumps = mumps
    self.mumps_ls = None

    LinearSolver.__init__(self, domain=domain, var=var, comm=comm, scheme=scheme, reordering=reordering, solver_name=LinearSolver.SolverMumps)


    self.system = system

    self._dim = self.domain.dim
    self.var = var
    self.domain.solver = "mumps"
    self.reordering = reordering
    self.reuse_mtx = reuse_mtx
    self.with_mtx = with_mtx
    self.memory_relaxation = memory_relaxation
    self.sol = None

    self.rhs0 = np.zeros(self.globalsize, dtype=FLOAT_TYPE)

  def __call__(self, rhs: npt.NDArray[Union[np.float32, np.float64]]=None):

    if not self.reuse_mtx:
      self.clear()

    rhs00 = self.presolve(reuse_mtx=self.reuse_mtx, with_mtx=self.with_mtx)

    if rhs is not None:
      rhs += rhs00
    else:
      rhs = rhs00

    # Allocation size of rhs
    if self.comm.Get_rank() == 0:
      self.sol = rhs.copy()
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

  def presolve(self, reuse_mtx=False, with_mtx=False):
    if not reuse_mtx or self.mumps_ls is None:
      self.update_ghost_values()

      # assembly row, col , data, rhs(bc)
      # if not with_mtx:
      #   print("Assembly")
      self.assembly()
      if self.reordering and self.comm.Get_size() == 1:
        print("=>Reordering the matrix ...")
        self.reordering_matrix()

      rhs00 = self.comm.reduce(self.rhs0, op=MPI.SUM, root=0)

      if self.mumps_ls is None:
        mem_relax = self.memory_relaxation
        self.mumps_ls = self.mumps.MumpsSolver(verbose=self.verbose, system=self.system,
                                               mem_relax=mem_relax)
      # Fortran indexing
      self._row += 1
      self._col += 1

      self.mumps_ls.set_rcd_distributed(self._row, self._col,
                                        self._data,
                                        self.globalsize)

      self.mumps_ls.set_icntl(18, 3)
      self.mumps_ls.set_icntl(16, 1)
      # self.mumps_ls.set_icntl(14, 40)

      if self.comm.Get_rank() == 0:
        self.sol = rhs00.copy()

      # Analyse
      self.mumps_ls._mumps_call(job=1)
      # Factorization Phase
      self.mumps_ls._mumps_call(job=2)

      # Allocation size of rhs
      if self.comm.Get_rank() == 0:
        self.mumps_ls.set_rhs_centralized(self.sol)
      else:
        self.sol = np.zeros(self.globalsize, dtype=FLOAT_TYPE)

      return rhs00
    return None

  def clear(self):
    self.mumps_ls = None
