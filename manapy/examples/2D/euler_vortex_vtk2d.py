#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isentropic vortex advection with VTK output of the exact AND simulated solution.

The isentropic vortex is a smooth *exact* solution of the 2D Euler equations: a
vortex superimposed on a uniform stream advects at the stream velocity without
any change of shape. This example advects it and, at every output step, writes a
VTK snapshot containing both the simulated fields and the exact fields (plus the
pointwise error), so the two can be overlaid/compared in ParaView.

VTK series (rho, rho_exact, rho_err, P, P_exact, u, u_exact) are written to
./vtk_results/. Control the cadence with OUTPUT_EVERY (iterations).

Run:
    MESH_DIR=../../../meshes/geo python3 euler_vortex_vtk2d.py
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
xc, yc, vol = cells.center[:, 0], cells.center[:, 1], cells.volume

# --- isentropic vortex parameters ---
gamma = 1.4
uinf, vinf, beta = 1.0, 0.0, 5.0
x0, y0 = xc.mean(), yc.mean()


def exact(t):
  """Exact isentropic-vortex state (rho, u, v, p) at time t (periodic advection)."""
  xv, yv = x0 + uinf * t, y0 + vinf * t
  dx, dy = xc - xv, yc - yv
  r2 = dx * dx + dy * dy
  fac = beta / (2 * np.pi) * np.exp(0.5 * (1 - r2))
  u = uinf - dy * fac
  v = vinf + dx * fac
  T = 1.0 - (gamma - 1) * beta ** 2 / (8 * gamma * np.pi ** 2) * np.exp(1 - r2)
  rho = T ** (1.0 / (gamma - 1))
  p = rho ** gamma
  return rho, u, v, p


rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
r, u, v, p = exact(0.0)
rho.cell[:] = r; P.cell[:] = p
rhou.cell[:] = r * u; rhov.cell[:] = r * v
rhoE.cell[:] = p / (gamma - 1) + 0.5 * r * (u * u + v * v)

solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                     order=2, scheme="rusanov", bc="Neumann")

output_every = int(os.environ.get('OUTPUT_EVERY', 50))


def save_vtk(t, dt, niter, miter):
  re, ue, ve, pe = exact(t)
  u_sim = rhou.cell / rho.cell
  domain.save_on_cell_multi(
      ["rho", "rho_exact", "rho_err", "P", "P_exact", "u", "u_exact"],
      [rho.cell, re, np.abs(rho.cell - re), P.cell, pe, u_sim, ue],
      dt, t, niter, miter)


tfinal = float(os.environ.get('TFINAL', 0.3))
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

# final L2 error vs exact (interior, away from the non-periodic Neumann boundary)
re, ue, ve, pe = exact(t)
m = (np.abs(xc - x0) < 0.35) & (np.abs(yc - y0) < 0.35)
l2_rho = np.sqrt(np.sum(vol[m] * (rho.cell[m] - re[m]) ** 2) / np.sum(vol[m]))
if RANK == 0:
  print(f"isentropic vortex advected to t={t:.3f} (exact: shifted by uinf*t={uinf*t:.3f})")
  print(f"  L2(rho) vs exact (interior) = {l2_rho:.3e}")
  if output_every:
    print(f"  wrote {miter + 1} VTK snapshots to ./vtk_results/ "
          f"(fields: rho/rho_exact/rho_err, P/P_exact, u/u_exact)")
