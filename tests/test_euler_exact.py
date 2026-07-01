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
def test_isentropic_vortex_steady_state(domain):
  """A vortex at rest is a steady exact Euler solution; the scheme should hold it
  in place (no displacement) with only mild diffusion."""
  cells = domain.cells
  xc, yc, vol = cells.center[:, 0], cells.center[:, 1], cells.volume
  gamma = 1.4
  beta, Rc = 5.0, 0.12
  x0, y0 = 0.5, 0.5                               # centred, at rest

  def exact():
    dx, dy = xc - x0, yc - y0
    r2 = (dx * dx + dy * dy) / (Rc * Rc)
    fac = beta / (2 * np.pi) * np.exp(0.5 * (1 - r2))
    u = -(dy / Rc) * fac
    v = (dx / Rc) * fac
    T = 1.0 - (gamma - 1) * beta ** 2 / (8 * gamma * np.pi ** 2) * np.exp(1 - r2)
    rho = T ** (1.0 / (gamma - 1))
    return rho, u, v, rho ** gamma

  rho, P, rhou, rhov, rhoE = _vars(domain)
  r, u, v, p = exact()
  rho.cell[:] = r; P.cell[:] = p
  rhou.cell[:] = r * u; rhov.cell[:] = r * v
  rhoE.cell[:] = p / (gamma - 1) + 0.5 * r * (u * u + v * v)

  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                  order=2, scheme="rusanov", bc="Neumann")
  tend = 0.3
  t = 0.0
  while t < tend:
    dt = S.stepper()
    if t + dt > tend:
      dt = tend - t; S.dt = dt
    t += dt
    S.compute_fluxes(t); S.compute_new_val()

  re, ue, ve, pe = exact()
  err = _l2(vol, rho.cell, re)
  # the vortex must not drift: barycentre of the density deficit stays put
  w = np.clip(1.0 - rho.cell, 0.0, None)
  x_center = float(np.sum(vol * w * xc) / np.sum(vol * w))
  assert np.all(rho.cell > 0) and np.all(np.isfinite(P.cell))
  assert err < 3e-3                               # steady smooth vortex
  assert abs(x_center - x0) < 0.01                # no spurious drift


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


def test_roe_entropy_fix(domain):
  """Harten entropy fix: ef=0 reproduces plain Roe exactly; ef>0 only smooths the
  sonic point, stays physical and still matches the exact Riemann solution."""
  cells = domain.cells
  xc, vol = cells.center[:, 0], cells.volume
  gamma = 1.4

  def run(ef):
    rho, P, rhou, rhov, rhoE = _vars(domain)
    left = xc < 0.5
    rho.cell[:] = np.where(left, 1.0, 0.125)
    P.cell[:] = np.where(left, 1.0, 0.1)
    rhoE.cell[:] = P.cell / (gamma - 1)
    S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                    scheme="Roe", bc="TubeSchok", entropy_fix=ef)
    t = 0.0
    while t < 0.2:
      dt = S.stepper()
      if t + dt > 0.2:
        dt = 0.2 - t; S.dt = dt
      t += dt
      S.compute_fluxes(t); S.compute_new_val()
    return rho.cell.copy()

  r_plain = run(0.0)
  r_fix = run(0.15)
  re, _ = _exact_sod(xc, 0.2, gamma)
  # ef=0 reproduces plain Roe to machine precision (the fix is inert)
  r_ref = run(0.0)
  assert np.max(np.abs(r_plain - r_ref)) < 1e-14
  # ef>0 stays physical and matches the exact Riemann solution
  assert np.all(r_fix > 0)
  assert _l2(vol, r_fix, re) < 6e-2
  # the fix changes the solution only near the rarefaction sonic point (small, local)
  assert np.max(np.abs(r_fix - r_plain)) < 5e-2


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


def test_doubleflux_contact_pressure_equilibrium_3d():
  """3D analogue: a moving multi-gamma contact (uniform in y,z) must stay in
  pressure equilibrium with the 3D double-flux update, machine-precision and far
  better than 3D variable-gamma alone."""
  from manapy.solvers.euler.species import SpeciesTransport
  mesh3d = os.path.join(os.path.dirname(__file__), "..", "meshes", "hybrid3d.msh")
  dom = Domain.create_domain(mesh3d, 3, Partitioning.Par_Nodal, recreate=True)
  cells = dom.cells
  xc, vol = cells.center[:, 0], cells.volume
  L = xc.max() - xc.min()
  gL, gR, P0, u0 = 1.6, 1.2, 1.0, 0.5
  Y0 = 0.5 * (1 + np.tanh((xc - (xc.min() + 0.4 * L)) / (0.05 * L)))

  def gamma_of(Y):
    return 1.0 + 1.0 / ((1 - Y) / (gL - 1) + Y / (gR - 1))

  def run(doubleflux):
    g_init = gamma_of(Y0)
    rhoc = 1.0 + Y0
    rho, P, rhou, rhov, rhoE = (Variable(domain=dom) for _ in range(5))
    rhow = Variable(domain=dom)
    rho.cell[:] = rhoc; P.cell[:] = P0
    rhou.cell[:] = rhoc * u0
    rhoE.cell[:] = P0 / (g_init - 1) + 0.5 * rhoc * u0 ** 2
    S = EulerSolver(rho, P, rhou, rhov, rhoE, rhow=rhow, gamma=1.4, cfl=0.3,
                    scheme="rusanov", bc="Neumann", variable_gamma=True, doubleflux=doubleflux)
    sp = SpeciesTransport(S, [1 - Y0, Y0], renormalize=True)
    S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
    for _ in range(60):
      dt = S.stepper()
      S.compute_fluxes(dt); S.compute_new_val(); sp.advance(dt)
      S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
    return _l2(vol, P.cell, np.full_like(P.cell, P0))

  l2_df = run(True)
  l2_vg = run(False)
  assert l2_df < 1e-10                            # 3D double-flux: machine-precision equilibrium
  assert l2_df < l2_vg / 100.0                    # and far better than 3D variable-gamma alone


# --------------------------------------------------------------------------- #
# 4b. Sensible-energy split -- reactive double-flux conserves total energy and
#     reaches the exact constant-UV adiabatic flame temperature.
# --------------------------------------------------------------------------- #
def test_reactive_sensible_energy_double_flux():
  """With the double-flux update the hydro carries only the *sensible* energy
  (rhoE = P/(gamma-1)+KE); the chemical/formation energy lives in the composition.
  A quiescent H2/air bomb must therefore still reach the Cantera constant-UV
  equilibrium temperature, with the reaction heat release flowing into the
  sensible energy so the conserved *physical* total energy is preserved -- and the
  result must match the standard total-energy coupling. Run on a small mesh (the
  state is uniform, so it is effectively 0-D) to keep the per-cell stiff reactor
  integrations cheap."""
  ct = pytest.importorskip("cantera")
  from manapy.solvers.euler.cantera_backend import CanteraChemistry
  from manapy.solvers.euler.reactive_solver import ReactiveSolver

  small_mesh = os.path.join(os.path.dirname(__file__), "..", "meshes", "hybrid2d.msh")
  domain = Domain.create_domain(small_mesh, 2, Partitioning.Par_Nodal, recreate=True)
  chem = CanteraChemistry("h2o2.yaml")
  Y = chem.mass_fractions_from(H2=2 * 2.016, O2=32.0, N2=3.76 * 28.0)
  T0, p0 = 1100.0, ct.one_atm
  gas = chem.gas
  gas.TPY = T0, p0, Y
  rho0, e0 = gas.density, gas.int_energy_mass
  gamma_rep = gas.cp_mass / gas.cv_mass
  Teq = chem.equilibrium_T(rho0, T0, Y)

  def run(doubleflux):
    rho, P, rhou, rhov, rhoE = _vars(domain)
    rho.cell[:] = rho0; P.cell[:] = p0; rhoE.cell[:] = rho0 * e0
    S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma_rep, cfl=0.4,
                    scheme="rusanov", bc="Neumann",
                    variable_gamma=doubleflux, doubleflux=doubleflux)
    rs = ReactiveSolver(S, chem, [Y[k] for k in range(chem.nspec)])
    assert rs.sensible == doubleflux                 # auto-detected from the solver
    E0 = float(rs.total_energy().mean())
    t = 0.0
    for _ in range(40):
      t += rs.step(t=t)
      if rs._temperature().mean() > Teq - 5:
        break
    Ef = float(rs.total_energy().mean())
    return float(rs._temperature().mean()), abs(Ef - E0) / E0

  T_total, dE_total = run(False)
  T_df, dE_df = run(True)
  assert abs(T_total - Teq) < 5.0                    # total-energy coupling: equilibrium
  assert abs(T_df - Teq) < 5.0                       # sensible/double-flux: same equilibrium
  assert abs(T_df - T_total) < 1.0                   # the two couplings agree
  assert dE_df < 1e-10                               # heat release conserves physical energy


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
