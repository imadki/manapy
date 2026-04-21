#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from mpi4py import MPI
import numpy as np
from manapy.backends.types import FLOAT_TYPE
import manapy.solvers.advecdiff.fvm_utils_compute as fvm_utils_compute

from manapy.core.Variable import Variable


class AdvectionDiffusionSolver:
  _parameters = [('Dxx', float, 0., 0.,
                  'Diffusion in x direction'),
                 ('Dyy', float, 0., 0.,
                  'Diffusion in y direction'),
                 ('Dzz', float, 0., 0.,
                  'Diffusion in z direction'),
                 ('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .4, 0,
                  'cfl of the explicit scheme')
                 ]
  def __init__(self,
               var: Variable,
               vel: tuple[Variable, Variable]|tuple[Variable, Variable, Variable],
               Dxx: float = 0.0,
               Dyy: float = 0.0,
               Dzz: float = 0.0,
               dt: float = 0.0,
               order: int = 1,
               cfl: float = 0.
            ):

    self.var = var
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.comm = self.domain.halo_comm.graph_comm

    self.u = vel[0]
    self.v = vel[1]
    self.w = vel[2] if len(vel) == 3 else Variable(domain=self.domain)

    self.Dxx = Dxx
    self.Dyy = Dyy
    self.Dzz = Dzz
    self.dt = dt
    self.order = order
    self.cfl = cfl
    self.diffusion = True

    if self.Dxx == self.Dyy == self.Dzz == 0:
      self.diffusion = False

    self.var.__dict__["convective"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)
    self.var.__dict__["dissipative"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)
    self.var.__dict__["source"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)


    self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_2d
    if self.dim == 3:
      self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_3d
    self._explicitscheme_dissipative = fvm_utils_compute.explicitscheme_dissipative
    self._time_step = fvm_utils_compute.time_step
    self._update_new_value = fvm_utils_compute.update_new_value

  def explicit_convective(self):
    if self.order == 2:
      self.var.compute_cell_gradient()
    self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo, self.u.face,
                                    self.v.face, self.w.face,
                                    self.var.gradcellx, self.var.gradcelly, self.var.gradcellz, self.var.gradhalocellx,
                                    self.var.gradhalocelly, self.var.gradhalocellz, self.var.psi, self.var.psihalo,
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
                                    self.domain.faces.cellid, self.domain.faces.normal,
                                    self.domain.faces.halofid, self.domain.faces.name,
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces,
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, self.order, self.domain.faces.ghost_id)

  def explicit_dissipative(self):
    self.var.compute_face_gradient()
    self._explicitscheme_dissipative(self.var.gradfacex, self.var.gradfacey, self.var.gradfacez,
                                     self.domain.faces.cellid,
                                     self.domain.faces.normal, self.domain.faces.name, self.var.dissipative, self.Dxx,
                                     self.Dyy, self.Dzz)

  def stepper(self):
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal,
                          self.domain.faces.mesure,
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim, self.Dxx, self.Dyy, self.Dzz)
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return self.dt

  def compute_fluxes(self):
    # interpolate cell to node
    self.var.update_halo_value()
    self.var.update_ghost_value()

    # convective flux
    self.explicit_convective()

    # dissipative flux
    if self.diffusion:
      self.var.interpolate_celltonode()
      self.explicit_dissipative()

  def compute_new_val(self):
    self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt,
                           self.domain.cells.volume)







