#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.compute import DiffusionSolverCompute


class DiffusionSolver:
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
    self.config = self.domain.config
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

    # Use add_term so these become GPUArray under the GPU backend (a raw np.zeros
    # would be a plain ndarray -> the dissipative kernel's writes would land on a
    # throwaway device copy and update_new_value would read zeros).
    self.var.add_term("convective")
    self.var.add_term("dissipative")
    self.var.add_term("source")

    # Resolves the device kernels once, here.
    self.compute = DiffusionSolverCompute(self.config, self.dim)
    self._explicitscheme_dissipative = self.compute.explicitscheme_dissipative
    self._time_step = self.compute.time_step
    self._update_new_value = self.compute.update_new_value

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

    # dissipative flux
    if self.diffusion:
      self.var.interpolate_celltonode()
      self.explicit_dissipative()

  def compute_new_val(self):
    self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt,
                           self.domain.cells.volume)
