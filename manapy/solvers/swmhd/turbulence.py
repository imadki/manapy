#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-epsilon RANS turbulent Shallow-Water MHD solver.

Composes the inviscid SRNH `ShallowWaterMHDSolver` (hyperbolic part: h, hu, hv,
hB1, hB2 with GLM cleaning and the well-balanced topography source) with the
Favre-averaged k-epsilon closure of "RANS modeling of SWMHD" (dual kinetic +
magnetic turbulent energy/dissipation):

  extra transported densities: kc=h*k_c, km=h*k_m, epsc=h*eps_c, epsm=h*eps_m
  eddy viscosities: nu_t = Cmu k_c^2/eps_c ,  mu_t = Cmu k_m^2/eps_m

Per step:
  1. derived Favre fields u~,v~,B1~,B2~ and their cell gradients
  2. eddy viscosities nu_t, mu_t (with realizability floors)
  3. SRNH convective flux + well-balanced source (reused ShallowWaterMHDSolver)
  4. turbulent momentum stress  div((nu+nu_t) h grad u~)  added to hu, hv
  5. k-epsilon transport: upwind advection + variable-coefficient diffusion
     div((nu+nu_t/sigma_k) h grad k) + algebraic production-dissipation source
  6. explicit update of all fields

Reuses manapy FV machinery throughout (SRNH flux, Variable cell/face gradients,
the advecdiff-style diffusion operator generalised to a variable coefficient in
turbulence_utils). VTK output via the domain save helpers, as usual.

Included: full anisotropic turbulent stress tensor S_u,S_v (kinetic Newtonian +
Maxwell magnetic part) and the turbulent+molecular resistivity div((mu+nu_t/sigma_k)
h grad B~) in the induction equations.

Scope still to be extended:
  - the kinetic<->magnetic cross-transfer source terms T_k, R_k1, R_k2, D_k (eq. 2.13);
  - the induction resistivity uses the gradient-diffusion (Laplacian) form; the model
    writes it in curl form -- equivalent for the mean-field diffusion, div-B handled
    by GLM cleaning;
  - order-2 transport (currently first-order upwind) and quantitative validation.
"""
from mpi4py import MPI
import numpy as np

from manapy.core.Variable import Variable
from manapy.solvers.swmhd.system import ShallowWaterMHDSolver
import manapy.solvers.swmhd.turbulence_utils as turb


class TurbulentSWMHDSolver:
  def __init__(self, h, hvel, hB, kc, km, epsc, epsm,
               PSI=None, Z=None, nu=1e-3, mu=1e-3,
               order=1, cfl=0.6, grav=1.0, GLM=100,
               Cmu=0.09, Ce1=1.3, Ce2=1.8, be1=1.3, be2=1.8, Ce3=1.26, be3=1.26,
               sigma_k=1.0, sigma_e=1.3, k_floor=1e-8, eps_floor=1e-8):
    self.h = h
    self.hu, self.hv = hvel
    self.hB1, self.hB2 = hB
    self.kc, self.km, self.epsc, self.epsm = kc, km, epsc, epsm
    self.domain = h.domain
    self.comm = self.domain.halo_comm.graph_comm

    # inviscid SRNH SWMHD core (reused verbatim)
    self.core = ShallowWaterMHDSolver(h=h, hvel=hvel, hB=hB, PSI=PSI, Z=Z,
                                      order=order, cfl=cfl, grav=grav, GLM=GLM)

    self.nu, self.mu = nu, mu
    self.Cmu, self.Ce1, self.Ce2, self.be1, self.be2 = Cmu, Ce1, Ce2, be1, be2
    self.Ce3, self.be3 = Ce3, be3
    self.sigma_k, self.sigma_e = sigma_k, sigma_e
    self.k_floor, self.eps_floor = k_floor, eps_floor
    self.cfl = cfl

    # turbulence transport terms
    for v in (self.kc, self.km, self.epsc, self.epsm):
      v.add_term("convective"); v.add_term("dissipative"); v.add_term("source")

    dom = self.domain
    mk = lambda: Variable(domain=dom)
    self.ut, self.vt = mk(), mk()          # Favre velocity u~, v~
    self.B1t, self.B2t = mk(), mk()        # Favre field B1~, B2~
    self.nu_t, self.mu_t = mk(), mk()      # eddy viscosities (cell)
    self.gam = mk()                        # per-cell diffusion coefficient buffer
    self.B1t.add_term("dissipative"); self.B2t.add_term("dissipative")  # turbulent resistivity
    ncell = len(self.domain.cells.center)
    nface = len(self.domain.faces.cellid)
    self.stress_u = np.zeros(ncell)
    self.stress_v = np.zeros(ncell)
    self.nut_f = np.zeros(nface); self.mut_f = np.zeros(nface); self.h_f = np.zeros(nface)
    self.gam_face = np.zeros(len(self.domain.faces.cellid))

    turb.setup()
    self.dt = 0.0

  # ---- helpers -------------------------------------------------------------
  def _derived_fields(self):
    h = np.asarray(self.h.cell)
    self.ut.cell[:] = np.asarray(self.hu.cell) / h
    self.vt.cell[:] = np.asarray(self.hv.cell) / h
    self.B1t.cell[:] = np.asarray(self.hB1.cell) / h
    self.B2t.cell[:] = np.asarray(self.hB2.cell) / h
    for v in (self.ut, self.vt, self.B1t, self.B2t):
      v.update_halo_value(); v.update_ghost_value()
      v.compute_cell_gradient()                       # for turbulence production source
      v.interpolate_celltonode(); v.compute_face_gradient()   # for the stress/resistivity fluxes

  def _eddy_viscosity(self):
    turb.eddy_viscosity(self.h.cell, self.kc.cell, self.epsc.cell, self.km.cell, self.epsm.cell,
                        self.nu_t.cell, self.mu_t.cell, self.Cmu, self.k_floor, self.eps_floor)
    for v in (self.nu_t, self.mu_t):
      v.update_halo_value(); v.update_ghost_value()

  def _diffuse(self, w, coef_cell):
    # div(coef grad w): coef given per cell -> interpolate to faces -> var-coef diffusion
    self.gam.cell[:] = coef_cell
    self.gam.update_halo_value(); self.gam.update_ghost_value()
    turb.cell_to_face_coef(self.gam.cell, self.gam.ghost, self.gam.halo,
                           self.domain.faces.cellid, self.domain.faces.name,
                           self.domain.faces.halofid, self.gam_face)
    w.update_halo_value(); w.update_ghost_value(); w.interpolate_celltonode(); w.compute_face_gradient()
    turb.diffusion_varcoef(w.gradfacex, w.gradfacey, w.gradfacez, self.gam_face,
                           self.domain.faces.cellid, self.domain.faces.normal,
                           self.domain.faces.name, w.dissipative)

  def _advect(self, w):
    w.update_halo_value(); w.update_ghost_value()
    turb.advect_density(w.cell, w.ghost, w.halo,
                        self.ut.cell, self.vt.cell, self.ut.ghost, self.vt.ghost,
                        self.ut.halo, self.vt.halo,
                        self.domain.faces.cellid, self.domain.faces.normal,
                        self.domain.faces.name, self.domain.faces.halofid, w.convective)

  # ---- main step -----------------------------------------------------------
  def step(self):
    h = np.asarray(self.h.cell)
    self._derived_fields()
    self._eddy_viscosity()

    # time step: hyperbolic CFL (core) capped by a turbulent-diffusion limit
    dt = self.core.stepper()
    diffmax = max(float((self.nu + np.asarray(self.nu_t.cell)).max()),
                  float((self.mu + np.asarray(self.nu_t.cell) / self.sigma_k).max()))
    diffmax = self.comm.allreduce(diffmax, op=MPI.MAX)
    vol = np.asarray(self.domain.cells.volume)
    dt_diff = 0.3 * float(vol.min()) / (diffmax + 1e-30)
    self.dt = self.comm.allreduce(min(dt, dt_diff), op=MPI.MIN)

    # (3) SRNH convective + well-balanced source for the hyperbolic variables
    self.core.update_halo_values()
    self.core.update_ghost_values()
    self.core.update_cpsi_global()
    self.core.explicit_convective()
    self.core.update_term_source()

    # (4) full anisotropic turbulent stress tensor (kinetic + Maxwell) -> hu, hv,
    #     evaluated directly at faces from face gradients (exact for linear fields).
    fcid = self.domain.faces.cellid; fname = self.domain.faces.name
    fnorm = self.domain.faces.normal; fhalo = self.domain.faces.halofid
    turb.cell_to_face_coef(self.nu_t.cell, self.nu_t.ghost, self.nu_t.halo, fcid, fname, fhalo, self.nut_f)
    turb.cell_to_face_coef(self.mu_t.cell, self.mu_t.ghost, self.mu_t.halo, fcid, fname, fhalo, self.mut_f)
    turb.cell_to_face_coef(self.h.cell, self.h.ghost, self.h.halo, fcid, fname, fhalo, self.h_f)
    turb.stress_divergence_face(self.ut.gradfacex, self.ut.gradfacey, self.vt.gradfacex, self.vt.gradfacey,
                                self.B1t.gradfacex, self.B1t.gradfacey, self.B2t.gradfacex, self.B2t.gradfacey,
                                self.nut_f, self.mut_f, self.h_f, self.nu, self.mu,
                                fcid, fnorm, fname, self.stress_u, self.stress_v)

    # (4b) turbulent + molecular resistivity  div((mu + nu_t/sigma_k) h grad B~) -> hB1, hB2
    etah = (self.mu + np.asarray(self.nu_t.cell) / self.sigma_k) * h
    self._diffuse(self.B1t, etah)
    resB1 = np.asarray(self.B1t.dissipative).copy()
    self._diffuse(self.B2t, etah)
    resB2 = np.asarray(self.B2t.dissipative).copy()

    # (5) k-epsilon transport: advection + diffusion + source
    self._advect(self.kc); self._advect(self.km)
    self._advect(self.epsc); self._advect(self.epsm)
    self._diffuse(self.kc, (self.nu + np.asarray(self.nu_t.cell) / self.sigma_k) * h)
    dkc = np.asarray(self.kc.dissipative).copy()
    self._diffuse(self.km, (self.mu + np.asarray(self.mu_t.cell) / self.sigma_k) * h)
    dkm = np.asarray(self.km.dissipative).copy()
    self._diffuse(self.epsc, (self.nu + np.asarray(self.nu_t.cell) / self.sigma_e) * h)
    dec = np.asarray(self.epsc.dissipative).copy()
    self._diffuse(self.epsm, (self.mu + np.asarray(self.mu_t.cell) / self.sigma_e) * h)
    dem = np.asarray(self.epsm.dissipative).copy()
    turb.turbulence_source(self.h.cell, self.kc.cell, self.km.cell, self.epsc.cell, self.epsm.cell,
                           self.nu_t.cell, self.mu_t.cell,
                           self.ut.gradcellx, self.ut.gradcelly, self.vt.gradcellx, self.vt.gradcelly,
                           self.B1t.gradcellx, self.B1t.gradcelly, self.B2t.gradcellx, self.B2t.gradcelly,
                           self.kc.source, self.km.source, self.epsc.source, self.epsm.source,
                           self.Cmu, self.Ce1, self.Ce2, self.be1, self.be2,
                           self.Ce3, self.be3, self.sigma_k, self.k_floor, self.eps_floor)

    # (6) explicit update
    dt = self.dt
    self.core.dt = dt
    self.core.compute_new_val()                      # h, hu, hv, hB1, hB2 (+GLM)
    hu = np.asarray(self.hu.cell); hv = np.asarray(self.hv.cell)
    hu += dt * self.stress_u / vol
    hv += dt * self.stress_v / vol
    hB1 = np.asarray(self.hB1.cell); hB2 = np.asarray(self.hB2.cell)
    hB1 += dt * resB1 / vol
    hB2 += dt * resB2 / vol
    for w, diss in ((self.kc, dkc), (self.km, dkm), (self.epsc, dec), (self.epsm, dem)):
      wc = np.asarray(w.cell)
      wc += dt * ((np.asarray(w.convective) + diss) / vol + np.asarray(w.source))
      np.maximum(wc, 0.0, out=wc)                     # positivity of turbulent energies
    return dt
