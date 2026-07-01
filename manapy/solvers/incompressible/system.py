#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible Navier-Stokes by a face-flux-consistent projection (Chorin) method on
the unstructured collocated finite-volume grid -- the manapy analogue of OpenFOAM's
icoFoam (laminar, transient, single phase).

Per step (velocity u=(u,v), pressure P, kinematic viscosity nu, density rho):
  1. predictor  u* = u^n + dt (-conv + nu*diff)              (conv by the div-free flux)
  2. face flux  phi* = u*_face . S_f
  3. pressure   A P = -(rho/dt) sum_f phi*_f                  (two-point Laplacian)
  4. correct    phi = phi* - (dt/rho) a_f (P_N - P_P)         (divergence-free by design)
                u   = u*   - (dt/rho) grad(P)                 (cell reconstruction)

All three operators (divergence, pressure Laplacian, correction) share the same
two-point face coefficient a_f = area/dist, so the corrected face flux is exactly
divergence-free -- this is what makes the collocated method stable. Validated on the
lid-driven cavity vs Ghia et al. (1982). Serial (direct sparse factorisation of A);
an MPI/PETSc assembly of the same operator is the next step.
"""
import numpy as np

from manapy.solvers.incompressible.fvm_utils_compute import get_kernels


class IncompressibleSolver:

  def __init__(self, u, v, P, nu=1e-2, rho=1.0, cfl=0.4, ncorr=2, u_bc=None, v_bc=None, poisson=None):
    """
    u, v, P : cell velocity / pressure Variables. P must carry BCs (Neumann on the
              walls + one Dirichlet reference so the pure-Neumann system is regular),
              e.g. Variable(domain, BC={..., 'bottom':'dirichlet'}, values_dict={'bottom':0}).
    nu, rho : kinematic viscosity, density.
    u_bc, v_bc : {boundary_name: wall velocity component} (default 0 on every wall).
    poisson : a manapy LinearSolver for the pressure Poisson -- the backend is the
              caller's *choice* (PETScKrylovSolver / MUMPSSolver / GinkgoDistributedSolver),
              built with scheme='fv' (the two-point cell Laplacian, consistent with the
              collocated correction and MPI-ready). If None, a PETSc CG is created.
    """
    self.u = u; self.v = v; self.P = P
    self.domain = dom = u.domain
    if u.dim != 2:
      raise NotImplementedError("IncompressibleSolver is wired for 2D")
    self.nu = float(nu); self.rho = float(rho); self.cfl = float(cfl)
    self.ncorr = int(ncorr)                            # PISO-style pressure correctors

    self.cellid = np.asarray(dom.faces.cellid, dtype=np.int64)
    self.fname = np.asarray(dom.faces.name, dtype=np.int64)
    self.normal = np.ascontiguousarray(np.asarray(dom.faces.normal)[:, :2])
    self.vol = np.asarray(dom.cells.volume)
    self.nc = dom.nbcells; self.nf = len(self.cellid)
    codes = {k: dom.BCs[k][1] for k in dom.BCs}

    # reuse manapy's FV face coefficient (fv_coeff = |Sf|^2/|Sf.d|); it is exactly the
    # coefficient the scheme='fv' pressure Laplacian assembles, so the divergence,
    # the Poisson and the correction share one operator.
    self.af = np.asarray(dom.faces.fv_coeff)
    self.uw = np.zeros(self.nf); self.vw = np.zeros(self.nf)
    for name, val in (u_bc or {}).items():
      self.uw[self.fname == codes[name]] = val
    for name, val in (v_bc or {}).items():
      self.vw[self.fname == codes[name]] = val
    self._is_int = self.fname == 0
    self._bnd = ~self._is_int

    # pressure Poisson: reuse manapy's two-point FV Laplacian (scheme='fv') through a
    # distributed linear solver -- backend is the caller's choice.
    if poisson is None:
      from manapy.solvers.ls import PETScKrylovSolver
      poisson = PETScKrylovSolver(domain=dom, var=P, reuse_mtx=True, scheme='fv',
                                  method="cg", precond="gamg", eps_a=1e-12, eps_r=1e-10)
    self.L = poisson

    self._face_flux, self._mom_rhs, self._gg_grad = get_kernels()
    self._phi = np.zeros(self.nf)
    self._du = np.zeros(self.nc); self._dv = np.zeros(self.nc)
    self._gx = np.zeros(self.nc); self._gy = np.zeros(self.nc)
    self._psign = 1.0                                  # pressure-Poisson RHS sign
    self.dt = 0.0

  def stepper(self):
    umax = max(np.max(np.abs(self.u.cell)), np.max(np.abs(self.v.cell)), 1e-12)
    h = np.sqrt(self.vol.min())
    dt_c = self.cfl * h / umax
    dt_d = self.cfl * h * h / (4.0 * self.nu) if self.nu > 0 else 1e30
    self.dt = min(dt_c, dt_d)
    return self.dt

  def _divergence(self, u, v):
    self._face_flux(u, v, self.uw, self.vw, self.normal, self.cellid, self.fname, self._phi)
    d = np.zeros(self.nc)
    np.add.at(d, self.cellid[self._is_int, 0], -self._phi[self._is_int])
    np.add.at(d, self.cellid[self._is_int, 1], self._phi[self._is_int])
    np.add.at(d, self.cellid[self._bnd, 0], -self._phi[self._bnd])
    return d

  def _cell_divergence(self, u, v):
    """Per-cell velocity divergence (1/vol) sum_f phi_f from the face fluxes."""
    self._face_flux(u, v, self.uw, self.vw, self.normal, self.cellid, self.fname, self._phi)
    d = np.zeros(self.nc)
    np.add.at(d, self.cellid[self._is_int, 0], self._phi[self._is_int])
    np.add.at(d, self.cellid[self._is_int, 1], -self._phi[self._is_int])
    np.add.at(d, self.cellid[self._bnd, 0], self._phi[self._bnd])
    return d / self.vol

  def step(self, dt=None):
    dt = self.stepper() if dt is None else dt
    u, v = self.u.cell, self.v.cell
    ff, mom, gg = self._face_flux, self._mom_rhs, self._gg_grad

    # 1-2. predictor (momentum convection by the div-free face flux + diffusion)
    ff(u, v, self.uw, self.vw, self.normal, self.cellid, self.fname, self._phi)
    mom(u, v, self._phi, self.af, self.uw, self.vw, self.cellid, self.fname, self.vol,
        self.nu, self._du, self._dv)
    uc = u + dt * self._du; vc = v + dt * self._dv

    # 3-4. PISO-style correctors: each solves a pressure (correction) from the current
    #      velocity divergence and applies the gradient correction. Iterating drives
    #      the residual collocated cell divergence down; the pressures accumulate.
    Ptot = np.zeros(self.nc)
    for _ in range(self.ncorr):
      div = self._cell_divergence(uc, vc)
      self.L(rhs=self._psign * (self.rho / dt) * div)  # solves into P.cell
      self.P.update_halo_value(); self.P.update_ghost_value()
      Ptot += self.P.cell
      gg(self.P.cell, self.normal, self.cellid, self.fname, self.vol, self._gx, self._gy)
      uc = uc - (dt / self.rho) * self._gx
      vc = vc - (dt / self.rho) * self._gy
    u[:] = uc; v[:] = vc; self.P.cell[:] = Ptot
    return dt

  def divergence_norm(self):
    """L2 norm of the discrete velocity divergence (from the face fluxes)."""
    d = self._cell_divergence(self.u.cell, self.v.cell)
    return float(np.sqrt(np.sum(d * d * self.vol)))
