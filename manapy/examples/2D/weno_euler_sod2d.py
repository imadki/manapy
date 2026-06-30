#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WENO Euler solver on the Sod shock tube, with VTK of WENO vs exact Riemann.

Uses WenoEulerSolver (unstructured WENO reconstruction of the four conservative
variables + Rusanov flux + SSP-RK3). At each output step a VTK snapshot is written
with the simulated and exact density (+ error) so the captured shock/contact/
rarefaction can be compared to the analytic Riemann solution in ParaView. The WENO
solution is markedly sharper than first-order Rusanov (printed L2 errors).

Run (numba compiles the WENO build + kernels on first use):
    MESH_DIR=../../../meshes/geo python3 weno_euler_sod2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.weno_euler import WenoEulerSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), 'carre.msh')
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc, vol = cells.center[:, 0], np.asarray(cells.volume)
gamma = 1.4


def exact_sod(t):
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
    return f(p, rhoL, pL, cL) + f(p, rhoR, pR, cR)

  p = 0.5
  for _ in range(100):
    dp = 1e-6 * p
    pn = p - fp(p) / ((fp(p + dp) - fp(p - dp)) / (2 * dp))
    if abs(pn - p) < 1e-12 * p:
      p = pn; break
    p = max(pn, 1e-9)
  pstar = p
  ustar = 0.5 * (f(pstar, rhoR, pR, cR) - f(pstar, rhoL, pL, cL))
  if t <= 0:
    return np.where(xc < 0.5, rhoL, rhoR)
  rho = np.zeros_like(xc)
  for i, xi in enumerate(xc):
    s = (xi - 0.5) / t
    if s < ustar:
      rs = rhoL * (pstar / pL) ** (1 / g)
      cs = cL * (pstar / pL) ** ((g - 1) / (2 * g))
      if s < uL - cL:
        rho[i] = rhoL
      elif s > ustar - cs:
        rho[i] = rs
      else:
        cc = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
        rho[i] = rhoL * (cc / cL) ** (2 / (g - 1))
    else:
      rs = rhoR * ((pstar / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pR + 1))
      SR = uR + cR * np.sqrt((g + 1) / (2 * g) * pstar / pR + (g - 1) / (2 * g))
      rho[i] = rhoR if s > SR else rs
  return rho


rho = Variable(domain=domain); rhou = Variable(domain=domain)
rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
left = xc < 0.5
rho.cell[:] = np.where(left, 1.0, 0.125)
rhoE.cell[:] = np.where(left, 1.0, 0.1) / (gamma - 1)

bc = {k: "outflow" for k in ("in", "out", "upper", "bottom")}
solver = WenoEulerSolver(domain, rho.cell, rhou.cell, rhov.cell, rhoE.cell,
                         gamma=gamma, cfl=0.3, bc=bc)

output_every = int(os.environ.get('OUTPUT_EVERY', 40))


def save(t, dt, niter, miter):
  domain.save_on_cell_multi(["rho", "rho_exact", "rho_err"],
                            [rho.cell, exact_sod(t), np.abs(rho.cell - exact_sod(t))],
                            dt, t, niter, miter)


tfinal = float(os.environ.get('TFINAL', 0.2))
t = 0.0
niter = 0
miter = 0
if output_every:
  save(0.0, 0.0, 0, miter); miter += 1
while t < tfinal:
  dt = solver.stepper()
  if t + dt > tfinal:
    dt = tfinal - t
  solver.step(dt)
  t += dt
  niter += 1
  if output_every and niter % output_every == 0:
    save(t, dt, niter, miter); miter += 1
if output_every:
  save(t, dt, niter, miter)

re = exact_sod(t)
l2 = np.sqrt(np.sum(vol * (rho.cell - re) ** 2) / np.sum(vol))
if RANK == 0:
  print(f"WENO-Euler Sod at t={t:.3f} ({niter} SSP-RK3 steps)")
  print(f"  L2(rho) vs exact Riemann = {l2:.3e}")
  if output_every:
    print(f"  wrote {miter + 1} VTK snapshots to ./vtk_results/ (rho, rho_exact, rho_err)")
