#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shallow-Water Magnetohydrodynamics (SWMHD) finite-volume solver.

Ported from the legacy `manapy/models/SWMHDModel/core_SWMHD.py` to the new
solver architecture, mirroring `manapy/solvers/shallowater/system.py`.

Conserved variables: h, hu, hv, hB1, hB2, PSI, Z.
  - (h)            : water height
  - (hu, hv)       : momentum
  - (hB1, hB2)     : magnetic field * height
  - PSI            : GLM potential for div(hB) = 0 cleaning
  - Z              : bottom topography

Divergence cleaning uses a GLM approach with a global cleaning speed `cpsi`.

Gravity: `grav` is a single solver parameter passed consistently to BOTH the
convective SRNH fluxes and the well-balanced topography source term. This is
required for the C-property (lake-at-rest preservation) demonstrated in
Cissé, Elmahi, Kissami & Ratnani, "A well-balanced finite volume solver for the
2D shallow water magnetohydrodynamic equations with topography", Comput. Phys.
Commun. (2024). (The very first legacy port hard-coded grav=1.0 in the flux and
grav=9.81 in the source, which breaks well-balancedness; that is fixed here.)
"""
from mpi4py import MPI
import manapy.solvers.swmhd.fvm_utils_compute as fvm_utils_compute
from manapy.core.Variable import Variable


class ShallowWaterMHDSolver:
  _parameters = [('dt', float, 0., 0.,
                  'time step'),
                 ('order', int, 1, 1,
                  'order of the convective scheme'),
                 ('cfl', float, .8, 0,
                  'cfl of the explicit scheme'),
                 ('grav', float, 1.0, 0.,
                  'gravity constant (convective part)'),
                 ('GLM', int, 100, 0,
                  'GLM divergence-cleaning switch (10 enables PSI relaxation)'),
                 ('f0', float, 0., 0.,
                  'beta-plane reference Coriolis parameter (0 disables rotation)'),
                 ('beta', float, 0., 0.,
                  'meridional gradient of the Coriolis parameter (Rossby beta); '
                  'df/dy = beta is what makes (magneto-)Rossby waves exist'),
                 ('y0', float, 0., 0.,
                  'beta-plane reference latitude, f = f0 + beta (y - y0)'),
                 ]

  def __init__(
          self,
          h: Variable,
          hvel: tuple[Variable, Variable],
          hB: tuple[Variable, Variable],
          PSI: Variable = None,
          Z: Variable = None,
          dt: float = 0.0,
          order: int = 1,
          cfl: float = 0.8,
          grav: float = 1.0,
          GLM: int = 100,
          f0: float = 0.0,
          beta: float = 0.0,
          y0: float = 0.0,
  ):

    self.h = h
    self.domain = self.h.domain
    self.dim = self.domain.dim
    self.halo_comm = self.domain.halo_comm

    self.hu = hvel[0]
    self.hv = hvel[1]
    self.hB1 = hB[0]
    self.hB2 = hB[1]

    if PSI is None:
      PSI = Variable(domain=self.domain)
    if Z is None:
      Z = Variable(domain=self.domain)

    self.PSI = PSI
    self.Z = Z
    self.varbs = {'h': self.h, 'hu': self.hu, 'hv': self.hv,
                  'hB1': self.hB1, 'hB2': self.hB2, 'PSI': self.PSI, 'Z': self.Z}

    terms = ['source', 'convective']
    for var in self.varbs.values():
      for term in terms:
        var.add_term(term)

    # Constants
    self.dt = dt
    self.order = order
    self.cfl = cfl
    self.grav = grav
    self.GLM = GLM
    self.f0 = f0
    self.beta = beta
    self.y0 = y0
    self.cpsi = 0.

    fvm_utils_compute.setup(self.dim)
    self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_SWMHD
    self._time_step_SWMHD = fvm_utils_compute.time_step_SWMHD
    self._cpsi_global = fvm_utils_compute.cpsi_global
    self._update_new_value = fvm_utils_compute.update_SWMHD
    self._term_source_srnh = fvm_utils_compute.term_source_srnh_SWMHD
    self._coriolis_source = fvm_utils_compute.coriolis_source_SWMHD

  def update_cpsi_global(self):
    cpsi = self._cpsi_global(self.h.cell, self.hu.cell, self.hv.cell, self.hB1.cell, self.hB2.cell,
                             self.cfl, self.domain.faces.normal, self.domain.faces.mesure,
                             self.domain.cells.volume, self.domain.cells.faceid)
    self.cpsi = self.halo_comm.graph_comm.allreduce(cpsi, MPI.MAX)
    return self.cpsi

  def explicit_convective(self):

    if self.order == 2:
      self.h.compute_cell_gradient()

    self._explicitscheme_convective(self.h.convective, self.hu.convective, self.hv.convective, self.hB1.convective,
                                    self.hB2.convective, self.PSI.convective, self.Z.convective,
                                    self.h.cell, self.hu.cell, self.hv.cell, self.hB1.cell, self.hB2.cell,
                                    self.PSI.cell, self.Z.cell,
                                    self.h.ghost, self.hu.ghost, self.hv.ghost, self.hB1.ghost, self.hB2.ghost,
                                    self.PSI.ghost, self.Z.ghost,
                                    self.h.halo, self.hu.halo, self.hv.halo, self.hB1.halo, self.hB2.halo,
                                    self.PSI.halo, self.Z.halo,
                                    self.h.gradcellx, self.h.gradcelly, self.h.gradhalocellx, self.h.gradhalocelly,
                                    self.h.psi, self.h.psihalo,
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
                                    self.domain.faces.cellid, self.domain.faces.mesure, self.domain.faces.normal,
                                    self.domain.faces.halofid,
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces,
                                    self.domain.periodicboundaryfaces,
                                    self.grav, self.order, self.cpsi)

  def stepper(self):
    ###### calculation of the time step
    dt_c = self._time_step_SWMHD(self.h.cell, self.hu.cell, self.hv.cell, self.hB1.cell, self.hB2.cell,
                                 self.cfl, self.domain.faces.normal, self.domain.faces.mesure,
                                 self.domain.cells.volume, self.domain.cells.faceid)

    self.dt = self.halo_comm.graph_comm.allreduce(dt_c, MPI.MIN)
    return self.dt

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
    self._term_source_srnh(self.h.source, self.hu.source, self.hv.source, self.hB1.source, self.hB2.source,
                           self.PSI.source, self.Z.source,
                           self.h.cell, self.Z.cell,
                           self.h.ghost, self.Z.ghost,
                           self.h.halo, self.Z.halo,
                           self.h.gradcellx, self.h.gradcelly, self.h.psi, self.h.gradhalocellx, self.h.gradhalocelly,
                           self.h.psihalo,
                           self.domain.cells.nodeid, self.domain.cells.faceid, self.domain.cells.cellfid,
                           self.domain.faces.cellid,
                           self.domain.cells.center, self.domain.cells.nf,
                           self.domain.faces.name, self.domain.faces.center, self.domain.halos.centvol,
                           self.domain.nodes.vertex, self.domain.faces.halofid, self.grav, self.order)

    # beta-plane Coriolis, ADDED after the well-balanced source (no-op if f0=beta=0)
    if self.f0 != 0. or self.beta != 0.:
      self._coriolis_source(self.hu.source, self.hv.source,
                            self.hu.cell, self.hv.cell,
                            self.domain.cells.center, self.domain.cells.volume,
                            self.f0, self.beta, self.y0)

  def compute_new_val(self):
    self._update_new_value(self.h.cell, self.hu.cell, self.hv.cell, self.hB1.cell, self.hB2.cell,
                           self.PSI.cell, self.Z.cell,
                           self.h.convective, self.hu.convective, self.hv.convective, self.hB1.convective,
                           self.hB2.convective, self.PSI.convective, self.Z.convective,
                           self.h.source, self.hu.source, self.hv.source, self.hB1.source, self.hB2.source,
                           self.PSI.source, self.Z.source,
                           self.dt, self.domain.cells.volume, self.GLM, self.cpsi)

  def compute_fluxes(self):

    # update halos
    requests = self.update_halo_values()

    # calculation of the time step
    self.stepper()

    MPI.Request.Waitall(requests)

    # update boundary conditions
    self.update_ghost_values()

    # global GLM cleaning speed (needed by the convective flux)
    self.update_cpsi_global()

    # convective flux
    self.explicit_convective()

    # topography source term
    self.update_term_source()
