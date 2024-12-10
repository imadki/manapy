from mpi4py import MPI
import numpy as np
from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
import manapy.solvers.advecdiff as solver_utils


class SolverFunctions():
  def __init__(self):
    self.explicitscheme_convective = solver_utils.explicitscheme_convective_3d
    self.explicitscheme_dissipative = solver_utils.explicitscheme_dissipative
    self.time_step = solver_utils.module.time_step
    self.update_new_value = solver_utils.module.update_new_value


class Solver():
  
  def __init__(self, cfl , Dxx, Dyy):

  
    
    


    running_conf = Struct(backend="numba", signature=True, cache=True, precision="double")
    MeshPartition("mesh.msh", dim=2, conf=running_conf, periodic=[0,0,0])

    running_conf = Struct(backend="numba", signature=True, cache =True, precision="double")
    domain = Domain(dim=2, conf=running_conf)

    ne = Variable(domain=domain)
    self.u  = Variable(domain=domain)
    self.v  = Variable(domain=domain)
    

    self.var = ne
    self.comm = self.var.comm
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.float_precision = self.domain.float_precision
    

    self.cfl   = np.float64(cfl)
    
    self.Dxx   = np.float64(Dxx)
    self.Dyy   = np.float64(Dyy)
    

    self.backend = self.domain.backend
    self.signature = self.domain.signature
    
    

    solvers = SolverFunctions()
    self._explicitscheme_convective  = self.backend.compile(solvers.explicitscheme_convective, signature=self.signature)
    self._explicitscheme_dissipative  = self.backend.compile(solvers.explicitscheme_dissipative, signature=self.signature)

 
    self._time_step  = self.backend.compile(solvers.time_step, signature=self.signature)
    self._update_new_value = self.backend.compile(solvers.update_new_value, signature=self.signature)
    self.dt    = self.__stepper()
    

    self.var.__dict__["convective"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    self.var.__dict__["dissipative"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    self.var.__dict__["source"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    

  


  
  def __explicit_convective(self):
    self.var.compute_cell_gradient()
    self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo, self.u.face, self.v.face, self.w.face,
                                    self.var.gradcellx, self.var.gradcelly, self.var.gradcellz, self.var.gradhalocellx, 
                                    self.var.gradhalocelly, self.var.gradhalocellz, self.var.psi, self.var.psihalo, 
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                    self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal, 
                                    self.domain.faces.halofid, self.domain.faces.name, 
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, order=self.order)
      
      

  
  def __explicit_dissipative(self):
    self.var.compute_face_gradient()
    self._explicitscheme_dissipative(self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.domain.faces.cellid, 
                                      self.domain.faces.normal, self.domain.faces.name, self.var.dissipative, self.Dxx, self.Dyy )   
      

  
  def __stepper(self):
    
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim, self.Dxx, self.Dyy )
      
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return  self.dt
    

  
  def compute_fluxes(self):

    self.var.update_halo_value()
    self.var.update_ghost_value()
    
    
    self.__explicit_convective()
      
    
    self.var.interpolate_celltonode()
    self.__explicit_dissipative()
      
    

      
  def compute_new_val(self):
      self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt, self.domain.cells.volume)
  

