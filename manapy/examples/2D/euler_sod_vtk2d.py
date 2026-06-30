#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sod shock tube with VTK output of the exact AND simulated solution.

Advances the standard Sod shock tube and, at each output step, writes a VTK
snapshot containing both the simulated fields and the exact Riemann solution
(rho, u, p) plus the pointwise density error -- so the captured shock, contact
and rarefaction can be compared against the analytic solution in ParaView.

VTK series written to ./vtk_results/. Cadence: OUTPUT_EVERY (iterations).

Run:
    MESH_DIR=../../../meshes/geo python3 euler_sod_vtk2d.py
    SCHEME=Roe ENTROPY_FIX=0.15 python3 euler_sod_vtk2d.py   # Harten entropy fix
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), 'carre.msh')
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc, vol = cells.center[:, 0], cells.volume

gamma = 1.4
x0 = 0.5
rhoL, uL, pL = 1.0, 0.0, 1.0
rhoR, uR, pR = 0.125, 0.0, 0.1


def exact_sod(t):
  """Exact Riemann solution (rho, u, p) of the Sod problem at time t."""
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
  if t <= 0:
    return (np.where(xc < x0, rhoL, rhoR), np.zeros_like(xc), np.where(xc < x0, pL, pR))
  rho = np.zeros_like(xc); u = np.zeros_like(xc); pr = np.zeros_like(xc)
  for i, xi in enumerate(xc):
    s = (xi - x0) / t
    if s < ustar:                                 # left of contact
      if pstar > pL:
        rs = rhoL * ((pstar / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pL + 1))
        SL = uL - cL * np.sqrt((g + 1) / (2 * g) * pstar / pL + (g - 1) / (2 * g))
        rho[i], u[i], pr[i] = (rhoL, uL, pL) if s < SL else (rs, ustar, pstar)
      else:
        rs = rhoL * (pstar / pL) ** (1 / g)
        cs = cL * (pstar / pL) ** ((g - 1) / (2 * g))
        SHL, STL = uL - cL, ustar - cs
        if s < SHL:
          rho[i], u[i], pr[i] = rhoL, uL, pL
        elif s > STL:
          rho[i], u[i], pr[i] = rs, ustar, pstar
        else:
          uu = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
          cc = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
          rho[i] = rhoL * (cc / cL) ** (2 / (g - 1)); u[i] = uu
          pr[i] = pL * (cc / cL) ** (2 * g / (g - 1))
    else:                                         # right of contact
      if pstar > pR:
        rs = rhoR * ((pstar / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pR + 1))
        SR = uR + cR * np.sqrt((g + 1) / (2 * g) * pstar / pR + (g - 1) / (2 * g))
        rho[i], u[i], pr[i] = (rhoR, uR, pR) if s > SR else (rs, ustar, pstar)
      else:
        rs = rhoR * (pstar / pR) ** (1 / g)
        cs = cR * (pstar / pR) ** ((g - 1) / (2 * g))
        SHR, STR = uR + cR, ustar + cs
        if s > SHR:
          rho[i], u[i], pr[i] = rhoR, uR, pR
        elif s < STR:
          rho[i], u[i], pr[i] = rs, ustar, pstar
        else:
          uu = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
          cc = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
          rho[i] = rhoR * (cc / cR) ** (2 / (g - 1)); u[i] = uu
          pr[i] = pR * (cc / cR) ** (2 * g / (g - 1))
  return rho, u, pr


rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
left = xc < x0
rho.cell[:] = np.where(left, rhoL, rhoR)
P.cell[:] = np.where(left, pL, pR)
rhoE.cell[:] = P.cell / (gamma - 1)

solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                     scheme=os.environ.get("SCHEME", "Roe"), bc="TubeSchok",
                     entropy_fix=float(os.environ.get("ENTROPY_FIX", 0.0)))

output_every = int(os.environ.get('OUTPUT_EVERY', 50))


def save_vtk(t, dt, niter, miter):
  re, ue, pe = exact_sod(t)
  u_sim = rhou.cell / rho.cell
  domain.save_on_cell_multi(
      ["rho", "rho_exact", "rho_err", "u", "u_exact", "P", "P_exact"],
      [rho.cell, re, np.abs(rho.cell - re), u_sim, ue, P.cell, pe],
      dt, t, niter, miter)


tfinal = float(os.environ.get('TFINAL', 0.2))
t = 0.0
niter = 0
miter = 0
if output_every:
  save_vtk(0.0, 0.0, 0, miter); miter += 1
while t < tfinal:
  dt = solver.stepper()
  if t + dt > tfinal:
    dt = tfinal - t; solver.dt = dt
  t += dt
  solver.compute_fluxes(t)
  solver.compute_new_val()
  niter += 1
  if output_every and niter % output_every == 0:
    save_vtk(t, dt, niter, miter); miter += 1
if output_every:
  save_vtk(t, dt, niter, miter)

re, ue, pe = exact_sod(t)
l2 = np.sqrt(np.sum(vol * (rho.cell - re) ** 2) / np.sum(vol))
if RANK == 0:
  print(f"Sod shock tube at t={t:.3f}  scheme={solver.scheme}")
  print(f"  L2(rho) vs exact Riemann = {l2:.3e}")
  if output_every:
    print(f"  wrote {miter + 1} VTK snapshots to ./vtk_results/ "
          f"(rho/rho_exact/rho_err, u/u_exact, P/P_exact)")
