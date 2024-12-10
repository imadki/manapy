from mpi4py import MPI
import numpy as np
from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
$$[__import_solver_utils]


class SolverFunctions():
  def __init__(self):
    $$[__advection_explicit_scheme]
    $$[__diffusion_explicit_scheme]
    self.time_step = solver_utils.module.time_step
    self.update_new_value = solver_utils.module.update_new_value


class Solver():
  
  def __init__(self, cfl $$[__diffusion_params]):

  
    $$[__init_variables]
    

    solvers = SolverFunctions()
    $$[__self_explicit_scheme_convective]
    $$[__self_explicit_scheme_dissipative]

 
    self._time_step  = self.backend.compile(solvers.time_step, signature=self.signature)
    self._update_new_value = self.backend.compile(solvers.update_new_value, signature=self.signature)
    self.dt    = self.__stepper()
    

    $$[__self_var_dict_convective]
    $$[__self_var_dict_dissipative]
    $$[__self_var_dict_source]
    

  


  $$[__def_explicit_convective]

  $$[__def_explicit_dissipative]

  $$[__def_stepper]

  $$[__def_compute_fluxes]

      
  def compute_new_val(self):
      self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt, self.domain.cells.volume)
  

