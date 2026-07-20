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
          scheme: str = "srnh",
          alphaf: float = 1.0,
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

    # --- numerical-flux scheme ------------------------------------------------
    self.scheme = str(scheme).lower()
    self.alphaf = alphaf
    if self.scheme not in ("srnh", "fvc"):
      raise ValueError(f"unknown scheme '{scheme}'; choose 'srnh' or 'fvc'")
    if self.scheme == "fvc":
      # Finite-Volume-Characteristics pipeline (eigenstructure-free, WB).
      self._node_value_for_interp = fvm_utils_compute.node_value_for_interpolation_2d
      self._departure = fvm_utils_compute.departure_SW_2d
      self._predictor = fvm_utils_compute.predictor_SW_2d
      self._explicitscheme_convective_fvc = fvm_utils_compute.explicitscheme_convective_SW_fvc
      self._init_fvc_workspace()

  def _init_fvc_workspace(self):
    # Helper primitive variables (for the Diamond face gradients feeding the
    # predictor) + the static 4-point interpolation-stencil geometry + work arrays.
    # NOTE: helper variables inherit the domain boundary types (BC=None); FVC thus
    # currently supports neumann / periodic domains (the far-field / open cases).
    dom = self.domain
    faces = dom.faces
    nbfaces = dom.nbfaces
    self.u = Variable(domain=dom)
    self.v = Variable(domain=dom)
    self.eta = Variable(domain=dom)
    nodeid = np.asarray(faces.nodeid)
    cellid = np.asarray(faces.cellid)
    name = np.asarray(faces.name)
    halofid = np.asarray(faces.halofid)
    vx = np.asarray(dom.nodes.vertex)
    cc = np.asarray(dom.cells.center)
    xC = np.zeros((nbfaces, 4))
    yC = np.zeros((nbfaces, 4))
    xC[:, 0] = vx[nodeid[:, 0], 0]; xC[:, 1] = vx[nodeid[:, 1], 0]
    yC[:, 0] = vx[nodeid[:, 0], 1]; yC[:, 1] = vx[nodeid[:, 1], 1]
    xC[:, 2] = cc[cellid[:, 0], 0]; yC[:, 2] = cc[cellid[:, 0], 1]
    inner = (name == 0); halo = (name == 10); bnd = ~(inner | halo)
    xC[inner, 3] = cc[cellid[inner, 1], 0]; yC[inner, 3] = cc[cellid[inner, 1], 1]
    if np.any(halo):
      hcv = np.asarray(dom.halos.centvol)
      xC[halo, 3] = hcv[halofid[halo], 0]; yC[halo, 3] = hcv[halofid[halo], 1]
    if np.any(bnd):
      gid = np.asarray(faces.ghost_id)
      gif = np.asarray(dom.ghost.info_flt)
      xC[bnd, 3] = gif[gid[bnd], 0]; yC[bnd, 3] = gif[gid[bnd], 1]
    self._xC = xC
    self._yC = yC
    z2 = lambda: np.zeros((nbfaces, 4))
    z1 = lambda: np.zeros(nbfaces)
    self._hVal = z2(); self._huVal = z2(); self._hvVal = z2(); self._hcVal = z2()
    self._X0 = z1(); self._Y0 = z1()
    self._h_p = z1(); self._hu_p = z1(); self._hv_p = z1(); self._hc_p = z1()

  def explicit_convective_fvc(self):
    # FVC: method-of-characteristics predictor + physical flux (well-balanced).
    dom = self.domain
    faces = dom.faces
    hsafe = np.maximum(self.h.cell, 1e-10)
    self.u.cell[:] = self.hu.cell / hsafe
    self.v.cell[:] = self.hv.cell / hsafe
    self.eta.cell[:] = self.h.cell + self.Z.cell
    # boundary/halo closure + Diamond face gradients of the primitive fields
    for var in (self.u, self.v, self.eta):
      if MPI.COMM_WORLD.size > 1:
        dom.halo_comm.exchange(var.cell, recv_buffer=var.halo)
      var.update_ghost_value()
      var.interpolate_celltonode()
      var.compute_face_gradient()
    # cell->node interpolation of the conservative variables -> ValForInterp stencil
    for var in (self.h, self.hu, self.hv, self.hc):
      var.interpolate_celltonode()
    self._node_value_for_interp(self._hVal, self.h.cell, self.h.node, self.h.ghost, self.h.halo,
                                faces.nodeid, faces.cellid, faces.halofid, faces.name)
    self._node_value_for_interp(self._huVal, self.hu.cell, self.hu.node, self.hu.ghost, self.hu.halo,
                                faces.nodeid, faces.cellid, faces.halofid, faces.name)
    self._node_value_for_interp(self._hvVal, self.hv.cell, self.hv.node, self.hv.ghost, self.hv.halo,
                                faces.nodeid, faces.cellid, faces.halofid, faces.name)
    self._node_value_for_interp(self._hcVal, self.hc.cell, self.hc.node, self.hc.ghost, self.hc.halo,
                                faces.nodeid, faces.cellid, faces.halofid, faces.name)
    self._departure(self._X0, self._Y0, self._hVal, self._huVal, self._hvVal,
                    self._xC, self._yC, faces.center, faces.normal, faces.mesure, self.dt, self.alphaf)
    self._predictor(self._h_p, self._hu_p, self._hv_p, self._hc_p,
                    self._hVal, self._huVal, self._hvVal, self._hcVal, self._xC, self._yC, self._X0, self._Y0,
                    self.u.gradfacex, self.u.gradfacey, self.v.gradfacex, self.v.gradfacey,
                    self.eta.gradfacex, self.eta.gradfacey, self.grav, self.dt, self.alphaf,
                    faces.normal, faces.mesure)
    self._explicitscheme_convective_fvc(self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective,
                                        self.Z.convective, self._h_p, self._hu_p, self._hv_p, self._hc_p,
                                        self.h.cell, self.Z.cell, self.h.ghost, self.Z.ghost, self.h.halo, self.Z.halo,
                                        faces.cellid, faces.mesure, faces.normal, faces.halofid,
                                        dom.innerfaces, dom.halofaces, dom.boundaryfaces, self.grav)

  def explicit_convective(self):

    if self.scheme == "fvc":
      self.explicit_convective_fvc()
      return

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

    # update term source (SRNH only; FVC folds the well-balanced pressure/bed
    # source directly into the convective residual via Audusse reconstruction)
    if self.scheme != "fvc":
      self.update_term_source()

    if self.fc != 0:
      # update coriolis forces
      self.update_term_coriolis()

    if self.wind:
      self.update_term_wind()

