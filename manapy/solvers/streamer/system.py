#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from mpi4py import MPI
import numpy as np

from manapy.backends.types import FLOAT_TYPE
from manapy.core.Variable import Variable
import manapy.solvers.advecdiff.fvm_utils_compute as advecdiff_fvm_utils_compute
import manapy.solvers.streamer.fvm_utils_compute as fvm_utils_compute


class StreamerSolver:
  _parameters = [('De', float, 0., 0.,
                  'Diffusion in x direction'),
                 ('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .4, 0,
                  'cfl of the explicit scheme'),
                 ('Mann', float, 0., 0.,
                  'Manning number for the friction'),
                 ('fc', float, 0., 0.,
                  'Coriolis force'),
                 ('grav', float, 9.81, 0.,
                  'gravity constant')
                 ]

  def __init__(self,
               ne: Variable,
               ni: Variable,
               vel: tuple[Variable, Variable] | tuple[Variable, Variable, Variable],
               E: tuple[Variable, Variable] | tuple[Variable, Variable, Variable],
               P: Variable,
               De: float = 0.0,
               dt: float = 0.0,
               order: int = 1,
               cfl: float = 0.4):

    self.ne = ne
    self.ni = ni
    self.domain = self.ne.domain
    self.dim = self.ne.dim
    self.comm = self.domain.halo_comm.graph_comm

    self.u = vel[0]
    self.v = vel[1]
    self.Ex = E[0]
    self.Ey = E[1]
    self.P = P
    self.w = vel[2] if len(vel) == 3 else Variable(domain=self.domain)
    self.Ez = E[2] if len(E) == 3 else Variable(domain=self.domain)
    self.vars = {'ne': self.ne, 'ni': self.ni}

    for var in self.vars.values():
      var.add_term('source')
      var.add_term('dissipation')
      var.add_term('convective')

    self.De = De
    self.dt = dt
    self.order = order
    self.cfl = cfl

    advecdiff_fvm_utils_compute.setup(self.dim)
    fvm_utils_compute.setup(self.dim)
    if self.dim == 2:
      self._explicitscheme_convective = advecdiff_fvm_utils_compute.explicitscheme_convective_2d
    elif self.dim == 3:
      self._explicitscheme_convective = advecdiff_fvm_utils_compute.explicitscheme_convective_3d

    self._explicitscheme_dissipative = fvm_utils_compute.explicitscheme_dissipative_ST
    self._explicitscheme_source = fvm_utils_compute.explicitscheme_source_ST
    self._compute_el_field = fvm_utils_compute.compute_el_field
    self._compute_velocity = fvm_utils_compute.compute_velocity
    self._time_step = fvm_utils_compute.time_step_ST
    self._update_new_value = fvm_utils_compute.update_ST
    self.update_rhs_glob = fvm_utils_compute.update_rhs_glob
    self.update_rhs_loc = fvm_utils_compute.update_rhs_loc

  def update_rhs(self):
    pass
    # TODO there is not solver attribute in domain
    # TODO there is not solver globalsize in domain
    # if self.domain.solver == "petsc":
    #   self.rhs_updated = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)
    #   self.update_rhs_loc(self.ne.cell, self.ni.cell, self.domain.cells.loctoglob, self.rhs_updated)
    # else:
    # domain.globalsize
    # self.rhs_updated = np.zeros(self.domain.globalsize, dtype=FLOAT_TYPE)
    # self.update_rhs_glob(self.ne.cell, self.ni.cell, self.domain.cells.loctoglob, self.rhs_updated)
    #
    # return self.rhs_updated

  def explicit_convective(self):
    if self.order == 2:
      self.ne.compute_cell_gradient()
    self._explicitscheme_convective(self.ne.convective, self.ne.cell, self.ne.ghost, self.ne.halo, self.u.face,
                                    self.v.face, self.w.face, self.ne.gradcellx, self.ne.gradcelly,
                                    self.ne.gradcellz, self.ne.gradhalocellx, self.ne.gradhalocelly,
                                    self.ne.gradhalocellz, self.ne.psi, self.ne.psihalo,
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
                                    self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal,
                                    self.domain.faces.halofid, self.domain.faces.name,
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces,
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, order=self.order)

  def explicit_dissipative(self):
    self.ne.compute_face_gradient()
    self._explicitscheme_dissipative(self.u.face, self.v.face, self.w.face, self.Ex.face, self.Ey.face, self.Ez.face,
                                     self.ne.gradfacex, self.ne.gradfacey, self.ne.gradfacez,
                                     self.domain.faces.cellid, self.domain.faces.normal, self.domain.faces.name,
                                     self.ne.dissipation)

  def stepper(self):
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell, self.Ez.cell, self.cfl,
                          self.domain.faces.normal, self.domain.faces.mesure, self.domain.cells.volume,
                          self.domain.cells.faceid, self.dim)
    self.dt = self.comm.allreduce(d_t, MPI.MIN)
    return self.dt

  def update_halo_values(self):
    requests = []
    if MPI.COMM_WORLD.size > 1:
      for var in self.vars.values():
        req = self.domain.halo_comm.immediate_exchange(var.cell, recv_buffer=var.halo)
        requests.append(req)
    return requests

  def update_ghost_values(self):
    for var in self.vars.values():
      var.update_ghost_value()

  def update_term_source(self, branching: int = 0):
    self._explicitscheme_source(self.ne.cell, self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell,
                                self.Ez.cell, self.ne.source, self.ni.source, self.domain.cells.center, branching)

  def compute_new_val(self):
    self._update_new_value(self.ne.cell, self.ni.cell, self.ne.convective, self.ni.convective,
                           self.ne.dissipation, self.ni.dissipation, self.ne.source, self.ni.source, self.dt,
                           self.domain.cells.volume)

  def compute_Electric_Field(self):
    self._compute_el_field(self.P.gradfacex, self.P.gradfacey, self.P.gradfacez, self.Ex.face, self.Ey.face,
                           self.Ez.face)

  def compute_Velocity(self):
    self._compute_velocity(self.Ex.face, self.Ey.face, self.Ez.face, self.u.face, self.v.face, self.w.face,
                           self.Ex.cell, self.Ey.cell, self.Ez.cell, self.u.cell, self.v.cell, self.w.cell,
                           self.domain.cells.faceid, self.dim)

  def compute_fluxes(self):
    requests = self.update_halo_values()
    if requests:
      MPI.Request.Waitall(requests)
    self.update_ghost_values()
    self.update_term_source()
    self.explicit_convective()
    self.ne.interpolate_celltonode()
    self.explicit_dissipative()
