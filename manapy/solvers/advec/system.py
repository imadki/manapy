#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""

from mpi4py import MPI
import numpy as np
from manapy.backends.types import FLOAT_TYPE
import manapy.solvers.advec.fvm_utils_compute as fvm_utils_compute

from manapy.core.Variable import Variable


class AdvectionSolver:
  _parameters = [('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .4, 0,
                  'cfl of the explicit scheme')
                 ]
  def __init__(self,
               var: Variable,
               vel: tuple[Variable, Variable]|tuple[Variable, Variable, Variable],
               dt: float = 0.0,
               order=1,
               cfl=0.8
          ):

    self.var = var
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.comm = self.domain.halo_comm.graph_comm

    self.u = vel[0]
    self.v = vel[1]
    self.w = vel[2] if len(vel) == 3 else Variable(domain=self.domain)
    self.dt = dt
    self.order = order
    self.cfl = cfl


    self.var.__dict__["convective"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)
    self.var.__dict__["dissipative"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)
    self.var.__dict__["source"] = np.zeros(self.domain.nbcells, dtype=FLOAT_TYPE)

    if self.dim == 2:
      self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_2d
    elif self.dim == 3:
      self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_3d

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

  def stepper(self):
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal,
                          self.domain.faces.mesure,
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim)
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return self.dt

  def compute_fluxes(self):

    # interpolate cell to node
    self.var.update_halo_value()
    self.var.update_ghost_value()

    # convective flux
    self.explicit_convective()

  def compute_new_val(self):
    self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt,
                           self.domain.cells.volume)







