#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from mpi4py import MPI
import manapy.solvers.shallowater.fvm_utils_compute as fvm_utils_compute
import manapy.solvers.advecdiff.fvm_utils_compute as advecdiff_fvm_utils_compute
from manapy.core.Variable import Variable
import numpy as np

class ShallowWaterSolver:
  _parameters = [('Dxx', float, 0., 0.,
                  'Diffusion in x direction'),
                 ('Dyy', float, 0., 0.,
                  'Diffusion in y direction'),
                 ('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .8, 0,
                  'cfl of the explicit scheme'),
                 ('Mann', float, 0., 0.,
                  'Manning number for the friction'),
                 ('fc', float, 0., 0.,
                  'Coriolis force'),
                 ('grav', float, 9.81, 0.,
                  'gravity constant'),
                 ('wind', bool, False, True,
                  'wind')
                 ]

  def __init__(
          self,
          h: Variable,
          hvel : tuple[Variable, Variable],
          hc: Variable,
          Z: Variable,
          Dxx: float = 0.0,
          Dyy: float = 0.0,
          dt: float = 0.0,
          order: int = 1,
          cfl: float = 0.8,
          Mann: float = 0,
          fc: float = 0,
          grav: float = 9.81,
          wind: bool = False,
  ):


    self.h = h
    self.domain = self.h.domain
    self.dim = self.h.dim
    self.halo_comm = self.domain.halo_comm

    self.hu = hvel[0]
    self.hv = hvel[1]

    if Z is None:
      Z = Variable(domain=self.domain)
    if hc is None:
      hc = Variable(domain=self.domain)

    self.hc = hc
    self.Z = Z
    self.varbs = {'h': self.h, 'hu': self.hu, 'hv': self.hv, 'hc': self.hc, 'Z': self.Z}

    terms = ['source', 'dissipation', 'coriolis', 'friction', "convective"]
    for var in self.varbs.values():
      for term in terms:
        var.add_term(term)

    # Constants
    self.Dxx = Dxx
    self.Dyy = Dyy
    self.Dzz = 0.
    self.dt = dt
    self.order = order
    self.cfl = cfl
    self.Mann = Mann
    self.fc = fc
    self.grav = grav
    self.wind = wind

    if self.Dxx == self.Dyy == 0:
      self.diffusion = False


    fvm_utils_compute.setup(self.dim)
    advecdiff_fvm_utils_compute.setup(self.dim)
    self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_SW
    self._explicitscheme_dissipative = advecdiff_fvm_utils_compute.explicitscheme_dissipative
    self._time_step_SW = fvm_utils_compute.time_step_SW
    self._update_new_value = fvm_utils_compute.update_SW
    self._term_coriolis_SW = fvm_utils_compute.term_coriolis_SW
    self._term_friction_SW = fvm_utils_compute.term_friction_SW
    self._term_wind_SW = fvm_utils_compute.term_wind_SW
    self._term_source_srnh = fvm_utils_compute.term_source_srnh_SW

  def explicit_convective(self):

    if self.order == 2:
      self.h.compute_cell_gradient()
      self.hc.compute_cell_gradient()

    self._explicitscheme_convective(self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective,
                                 self.Z.convective,
                                 self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                                 self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost, self.h.halo,
                                 self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                                 self.h.gradcellx, self.h.gradcelly, self.h.gradhalocellx, self.h.gradhalocelly,
                                 self.hc.gradcellx, self.hc.gradcelly, self.hc.gradhalocellx,
                                 self.hc.gradhalocelly, self.hc.psi, self.hc.psihalo,
                                 self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
                                 self.domain.faces.ghost_id, self.domain.ghost.info_flt, self.domain.faces.cellid, self.domain.faces.mesure,
                                 self.domain.faces.normal, self.domain.faces.halofid,
                                 self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, self.grav,
                                 self.order)

  def explicit_dissipative(self):

    self.hc.compute_face_gradient()
    self._explicitscheme_dissipative(self.hc.gradfacex, self.hc.gradfacey, self.hc.gradfacez, self.domain.faces.cellid,
                                     self.domain.faces.normal, self.domain.faces.name, self.hc.dissipation, self.Dxx,
                                     self.Dyy, self.Dzz)

  def stepper(self):
    ######calculation of the time step
    dt_c = self._time_step_SW(self.h.cell, self.hu.cell, self.hv.cell, self.cfl, self.domain.faces.normal,
                              self.domain.faces.mesure,
                              self.domain.cells.volume, self.domain.cells.faceid, self.grav, self.Dxx, self.Dyy)

    self.dt = self.halo_comm.graph_comm.allreduce(dt_c, MPI.MIN)
    return  self.dt

  def update_halo_values(self):
    requests = []
    if MPI.COMM_WORLD.size > 1:
      for var in self.varbs.values():
        req = self.halo_comm.immediate_exchange(var.cell, recv_buffer=var.halo)
        requests.append(req)
    return requests

  def update_ghost_values(self):
    for var in self.varbs.values():
      var.update_ghost_value()

  def interpolate_cell2node(self):
    for var in self.varbs.values():
      var.interpolate_celltonode()

  def update_term_source(self):
    self._term_source_srnh(self.h.source, self.hu.source, self.hv.source, self.hc.source, self.Z.source,
                           self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                           self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost,
                           self.h.halo, self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                           self.h.gradcellx, self.h.gradcelly, self.hc.psi, self.h.gradhalocellx, self.h.gradhalocelly,
                           self.hc.psihalo,
                           self.domain.cells.nodeid, self.domain.cells.faceid, self.domain.cells.cellfid,
                           self.domain.faces.cellid,
                           self.domain.cells.center, self.domain.cells.nf,
                           self.domain.faces.name, self.domain.faces.center, self.domain.halos.centvol,
                           self.domain.nodes.vertex, self.domain.faces.halofid, self.grav, self.order)

  def update_term_friction(self):
    self._term_friction_SW(self.h.cell, self.hu.cell, self.hv.cell, self.grav, self.Mann, self.dt)

  def update_term_coriolis(self):
    self._term_coriolis_SW(self.hu.cell, self.hv.cell, self.hu.coriolis, self.hv.coriolis, self.fc)

  def compute_new_val(self):
    self._update_new_value(self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                           self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective,
                           self.Z.convective,
                           self.h.source, self.hu.source, self.hv.source, self.hc.source, self.Z.source,
                           self.hc.dissipation, self.hu.coriolis, self.hv.coriolis,
                           0., 0., self.dt, self.domain.cells.volume)

  def update_term_wind(self):
    # TODO parameters ??
    self._term_wind_SW(self.domain.cells.center, self.Tx_wind, self.Ty_wind, self.wind, self.iteration)

  def compute_fluxes(self):

    # update halos
    requests = self.update_halo_values()

    # update friction term
    if self.Mann != 0:
      self.update_term_friction()

    self.stepper()

    MPI.Request.Waitall(requests)

    # update boundary conditions
    self.update_ghost_values()

    # convective flux
    self.explicit_convective()

    # dissipative flux
    if self.diffusion:
      # self.var.interpolate_celltonode()
      self.h.interpolate_celltonode()
      self.explicit_dissipative()

    # update term source
    self.update_term_source()

    if self.fc != 0:
      # update coriolis forces
      self.update_term_coriolis()

    if self.wind:
      self.update_term_wind()

