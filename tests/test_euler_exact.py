#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-solution tests for the compressible Euler/Navier-Stokes solver.

Each test checks the solver against a known analytic solution:
  * isentropic vortex      -- smooth exact Euler solution (advects unchanged);
  * Sod shock tube         -- exact Riemann solution (Roe sharper than Rusanov);
  * viscous shear decay    -- exact unsteady Navier-Stokes solution;
  * multi-gamma contact    -- exact pressure equilibrium (double-flux);
  * passive species advection -- exact conservation (sum of mass fractions, mass).
"""
import os
import numpy as np
import pytest

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver

MESH = os.path.join(os.path.dirname(__file__), "..", "meshes", "geo", "carre.msh")


@pytest.fixture(scope="module")
def domain():
  return Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)


def _vars(domain):
  return [Variable(domain=domain) for _ in range(5)]


def _l2(vol, num, exact):
  return float(np.sqrt(np.sum(vol * (num - exact) ** 2) / np.sum(vol)))


# --------------------------------------------------------------------------- #
# 1. Isentropic vortex -- smooth exact Euler solution that advects unchanged
# --------------------------------------------------------------------------- #
def test_isentropic_vortex_advects_unchanged(domain):
  cells = domain.cells
  xc, yc, vol = cells.center[:, 0], cells.center[:, 1], cells.volume
  gamma = 1.4
  uinf, vinf, beta = 1.0, 0.0, 5.0
  x0, y0 = xc.mean(), yc.mean()

  def exact(t):
    xv = x0 + uinf * t
    dx, dy = xc - xv, yc - y0
    r2 = dx * dx + dy * dy
    fac = beta / (2 * np.pi) * np.exp(0.5 * (1 - r2))
    u = uinf - dy * fac
    v = vinf + dx * fac
    T = 1.0 - (gamma - 1) * beta ** 2 / (8 * gamma * np.pi ** 2) * np.exp(1 - r2)
    rho = T ** (1.0 / (gamma - 1))
    p = rho ** gamma
    return rho, u, v, p

  rho, P, rhou, rhov, rhoE = _vars(domain)
  r, u, v, p = exact(0.0)
  rho.cell[:] = r; P.cell[:] = p
  rhou.cell[:] = r * u; rhov.cell[:] = r * v
  rhoE.cell[:] = p / (gamma - 1) + 0.5 * r * (u * u + v * v)

  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                  order=2, scheme="rusanov", bc="Neumann")
  tend = 0.15
  t = 0.0
  while t < tend:
    dt = S.stepper()
    if t + dt > tend:
      dt = tend - t; S.dt = dt
    t += dt
    S.compute_fluxes(t); S.compute_new_val()

  re, ue, ve, pe = exact(t)
  # interior only (Neumann boundary is not exactly the vortex far field)
  m = (np.abs(xc - x0) < 0.35) & (np.abs(yc - y0) < 0.35)
  err = _l2(vol[m], rho.cell[m], re[m])
  assert np.all(rho.cell > 0) and np.all(np.isfinite(P.cell))
  assert err < 1e-2                              # smooth solution, order-2 on a coarse mesh


# --------------------------------------------------------------------------- #
# 2. Sod shock tube vs the exact Riemann solution
# --------------------------------------------------------------------------- #
def _exact_sod(x, t, gamma=1.4, x0=0.5):
  rhoL, uL, pL = 1.0, 0.0, 1.0
  rhoR, uR, pR = 0.125, 0.0, 0.1
  g = gamma
  cL = np.sqrt(g * pL / rhoL); cR = np.sqrt(g * pR / rhoR)

  def f(p, rk, pk, ck):
    if p > pk:
      A = 2.0 / ((g + 1) * rk); B = (g - 1) / (g + 1) * pk
      return (p - pk) * np.sqrt(A / (p + B))
    return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

  def fp(p):
    return f(p, rhoL, pL, cL) + f(p, rhoR, pR, cR) + (uR - uL)

  p = 0.5 * (pL + pR)
  for _ in range(100):
    dp = 1e-6 * p
    der = (fp(p + dp) - fp(p - dp)) / (2 * dp)
    pn = p - fp(p) / der
    if abs(pn - p) < 1e-12 * p:
      p = pn; break
    p = max(pn, 1e-9)
  pstar = p
  ustar = 0.5 * (uL + uR) + 0.5 * (f(pstar, rhoR, pR, cR) - f(pstar, rhoL, pL, cL))
  rho = np.zeros_like(x); pr = np.zeros_like(x)
  for i, xi in enumerate(x):
    s = (xi - x0) / t
    if s < ustar:
      if pstar > pL:
        rs = rhoL * ((pstar / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pL + 1))
        SL = uL - cL * np.sqrt((g + 1) / (2 * g) * pstar / pL + (g - 1) / (2 * g))
        rho[i], pr[i] = (rhoL, pL) if s < SL else (rs, pstar)
      else:
        rs = rhoL * (pstar / pL) ** (1 / g)
        cs = cL * (pstar / pL) ** ((g - 1) / (2 * g))
        SHL, STL = uL - cL, ustar - cs
        if s < SHL:
          rho[i], pr[i] = rhoL, pL
        elif s > STL:
          rho[i], pr[i] = rs, pstar
        else:
          cc = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
          rho[i] = rhoL * (cc / cL) ** (2 / (g - 1)); pr[i] = pL * (cc / cL) ** (2 * g / (g - 1))
    else:
      if pstar > pR:
        rs = rhoR * ((pstar / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pR + 1))
        SR = uR + cR * np.sqrt((g + 1) / (2 * g) * pstar / pR + (g - 1) / (2 * g))
        rho[i], pr[i] = (rhoR, pR) if s > SR else (rs, pstar)
      else:
        rs = rhoR * (pstar / pR) ** (1 / g)
        cs = cR * (pstar / pR) ** ((g - 1) / (2 * g))
        SHR, STR = uR + cR, ustar + cs
        if s > SHR:
          rho[i], pr[i] = rhoR, pR
        elif s < STR:
          rho[i], pr[i] = rs, pstar
        else:
          cc = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
          rho[i] = rhoR * (cc / cR) ** (2 / (g - 1)); pr[i] = pR * (cc / cR) ** (2 * g / (g - 1))
  return rho, pr


@pytest.mark.parametrize("scheme", ["rusanov", "Roe"])
def test_sod_vs_exact_riemann(domain, scheme):
  cells = domain.cells
  xc, vol = cells.center[:, 0], cells.volume
  gamma = 1.4
  rho, P, rhou, rhov, rhoE = _vars(domain)
  left = xc < 0.5
  rho.cell[:] = np.where(left, 1.0, 0.125)
  P.cell[:] = np.where(left, 1.0, 0.1)
  rhoE.cell[:] = P.cell / (gamma - 1)
  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme=scheme, bc="TubeSchok")
  t, tend = 0.0, 0.2
  while t < tend:
    dt = S.stepper()
    if t + dt > tend:
      dt = tend - t; S.dt = dt
    t += dt
    S.compute_fluxes(t); S.compute_new_val()
  re, pe = _exact_sod(xc, tend, gamma)
  err_rho = _l2(vol, rho.cell, re)
  assert np.all(rho.cell > 0)
  assert err_rho < 6e-2                           # first order on a coarse mesh


def test_roe_is_sharper_than_rusanov(domain):
  cells = domain.cells
  xc, vol = cells.center[:, 0], cells.volume
  gamma = 1.4

  def run(scheme):
    rho, P, rhou, rhov, rhoE = _vars(domain)
    left = xc < 0.5
    rho.cell[:] = np.where(left, 1.0, 0.125)
    P.cell[:] = np.where(left, 1.0, 0.1)
    rhoE.cell[:] = P.cell / (gamma - 1)
    S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme=scheme, bc="TubeSchok")
    t = 0.0
    while t < 0.2:
      dt = S.stepper()
      if t + dt > 0.2:
        dt = 0.2 - t; S.dt = dt
      t += dt
      S.compute_fluxes(t); S.compute_new_val()
    re, _ = _exact_sod(xc, 0.2, gamma)
    return _l2(vol, rho.cell, re)

  assert run("Roe") < run("rusanov")


# --------------------------------------------------------------------------- #
# 3. Viscous shear-layer decay vs the exact unsteady Navier-Stokes solution
# --------------------------------------------------------------------------- #
def test_viscous_shear_decay_vs_exact(domain):
  cells = domain.cells
  yc, vol = cells.center[:, 1], cells.volume
  gamma, R, rho0, p0, U0, mu, Pr = 1.4, 287.0, 1.0, 10.0, 0.05, 0.1, 0.72
  nu = mu / rho0
  rho, P, rhou, rhov, rhoE = _vars(domain)

  def exact_u(t):
    return U0 * np.sin(np.pi * yc) * np.exp(-nu * np.pi ** 2 * t)

  rho.cell[:] = rho0; P.cell[:] = p0
  rhou.cell[:] = rho0 * exact_u(0.0); rhov.cell[:] = 0.0
  rhoE.cell[:] = p0 / (gamma - 1) + 0.5 * rhou.cell ** 2 / rho0
  bc_vel = {"in": "neumann", "out": "neumann", "upper": "dirichlet", "bottom": "dirichlet"}
  bc_temp = {k: "neumann" for k in ("in", "out", "upper", "bottom")}
  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.5, scheme="rusanov", bc="Neumann",
                  viscous=True, mu=mu, Pr=Pr, R=R, viscosity_law="constant", cfl_visc=0.3,
                  bc_vel=bc_vel, bc_temp=bc_temp, vel_values={"upper": 0.0, "bottom": 0.0})
  tau = 1.0 / (nu * np.pi ** 2)
  tend = 0.1 * tau
  t = 0.0
  while t < tend:
    dt = S.stepper()
    if t + dt > tend:
      dt = tend - t; S.dt = dt
    t += dt
    S.compute_fluxes(t); S.compute_new_val()
  err = _l2(vol, rhou.cell / rho.cell, exact_u(t))
  assert err < 5e-3                               # exact NS solution, structured-ish mesh


# --------------------------------------------------------------------------- #
# 4. Multi-gamma contact: double-flux keeps pressure exactly uniform
# --------------------------------------------------------------------------- #
def test_doubleflux_contact_pressure_equilibrium(domain):
  from manapy.solvers.euler.species import SpeciesTransport
  cells = domain.cells
  xc, vol = cells.center[:, 0], cells.volume
  gL, gR, P0, u0 = 1.6, 1.2, 1.0, 0.5
  Y0 = 0.5 * (1 + np.tanh((xc - 0.4) / 0.02))

  def gamma_of(Y):
    return 1.0 + 1.0 / ((1 - Y) / (gL - 1) + Y / (gR - 1))

  def run(doubleflux):
    g_init = gamma_of(Y0)
    rhoc = 1.0 + Y0
    rho, P, rhou, rhov, rhoE = _vars(domain)
    rho.cell[:] = rhoc; P.cell[:] = P0
    rhou.cell[:] = rhoc * u0; rhov.cell[:] = 0.0
    rhoE.cell[:] = P0 / (g_init - 1) + 0.5 * rhoc * u0 ** 2
    S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=1.4, cfl=0.3, scheme="rusanov",
                    bc="Neumann", variable_gamma=True, doubleflux=doubleflux)
    sp = SpeciesTransport(S, [1 - Y0, Y0], renormalize=True)
    S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
    t = 0.0
    while t < 0.2:
      dt = S.stepper()
      if t + dt > 0.2:
        dt = 0.2 - t; S.dt = dt
      t += dt
      S.compute_fluxes(t); S.compute_new_val(); sp.advance(dt)
      S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
    return _l2(vol, P.cell, np.full_like(P.cell, P0))

  l2_df = run(True)
  l2_vg = run(False)
  assert l2_df < 1e-10                            # double-flux: machine-precision equilibrium
  assert l2_df < l2_vg / 100.0                    # and far better than variable-gamma alone


# --------------------------------------------------------------------------- #
# 5. Passive species advection -- exact conservation
# --------------------------------------------------------------------------- #
def test_species_advection_conserves_sum_and_mass(domain):
  from manapy.solvers.euler.species import SpeciesTransport
  cells = domain.cells
  xc, vol = cells.center[:, 0], cells.volume
  gamma, rho0, p0, u0 = 1.4, 1.0, 10.0, 1.0
  rho, P, rhou, rhov, rhoE = _vars(domain)
  rho.cell[:] = rho0; P.cell[:] = p0; rhou.cell[:] = rho0 * u0
  rhoE.cell[:] = p0 / (gamma - 1) + 0.5 * rho0 * u0 ** 2
  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme="rusanov", bc="Neumann")
  L = xc.max() - xc.min()
  Y1 = 0.2 + 0.6 * np.exp(-((xc - (xc.min() + 0.3 * L)) / (0.08 * L)) ** 2)
  sp = SpeciesTransport(S, [Y1, 1 - Y1], renormalize=False)
  m0 = sp.total_mass(0) + sp.total_mass(1)
  dt = S.stepper()
  for _ in range(200):
    sp.advance(dt)
  Y = sp.mass_fractions()
  assert np.max(np.abs(Y[0] + Y[1] - 1.0)) < 1e-8           # sum of mass fractions stays 1
  m1 = sp.total_mass(0) + sp.total_mass(1)
  assert abs(m1 - m0) / m0 < 1e-12                          # total species mass conserved
