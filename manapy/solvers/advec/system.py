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
  # Numerical-flux schemes: name -> integer code used by the compute kernel.
  SCHEMES = ("upwind", "centered", "rusanov", "lax_friedrichs")

  _parameters = [('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .4, 0,
                  'cfl of the explicit scheme'),
                 ('scheme', str, 'upwind', 'upwind',
                  'numerical flux scheme (upwind or centered)')
                 ]
  def __init__(self,
               var: Variable,
               vel: tuple[Variable, Variable]|tuple[Variable, Variable, Variable],
               dt: float = 0.0,
               order=1,
               cfl=0.8,
               scheme="upwind"
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

    if scheme not in AdvectionSolver.SCHEMES:
      raise ValueError(f"unknown scheme '{scheme}'; choose from {list(AdvectionSolver.SCHEMES)}")
    self.scheme = scheme


    self.var.add_term("convective")
    self.var.add_term("dissipative")
    self.var.add_term("source")

    fvm_utils_compute.setup(self.dim, self.scheme)
    if self.domain.backend.name == "gpu":
      if self.dim != 2:
        raise NotImplementedError("Advection GPU is implemented for 2D only")
      from manapy.solvers.advec.cuda_fvm_utils import (
        get_kernel_explicitscheme_convective_2d,
        get_kernel_time_step,
        get_kernel_update_new_value,
      )
      self._explicitscheme_convective = get_kernel_explicitscheme_convective_2d()
      self._time_step = get_kernel_time_step()
      self._update_new_value = get_kernel_update_new_value()
    else:
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
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, self.order)

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







