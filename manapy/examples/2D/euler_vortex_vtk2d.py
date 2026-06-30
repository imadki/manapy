#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isentropic vortex with VTK output of the exact AND simulated solution.

The isentropic vortex is a smooth *exact* solution of the 2D Euler equations.
By default it is taken at rest (UINF=0): a **steady** exact solution that the
scheme should simply maintain, so the simulated and exact fields stay at the same
location and overlap in ParaView (only slight numerical diffusion differs).

Set UINF=1 to instead advect the vortex with a uniform stream -- still an exact
solution, but on a coarse mesh the first-order-in-time Rusanov scheme advects it
with a visible phase lag, so the simulated vortex trails the exact one. That
displacement (not amplitude error) is the dispersion of the scheme; it shrinks
with mesh/time refinement.

At every output step a VTK snapshot is written with both the simulated fields and
the exact fields (+ pointwise error) to ./vtk_results/. Cadence: OUTPUT_EVERY.

Run:
    MESH_DIR=../../../meshes/geo python3 euler_vortex_vtk2d.py     # steady (overlaps)
    UINF=1 python3 euler_vortex_vtk2d.py                           # advected (shows dispersion)
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
# The vortex must be small relative to the domain (here the unit square) so it
# stays localized and decays to the uniform free stream well before the boundary
# -- otherwise the boundary cuts the vortex and the "exact" comparison is spoiled.
# Rc is the core radius (~10 cells); the form below is an exact Euler solution.
gamma = 1.4
uinf = float(os.environ.get('UINF', 0.0))        # 0 -> steady vortex (overlaps exact)
vinf, beta = 0.0, 5.0
Rc = 0.12
# centre left when advecting so the vortex stays interior; centred when steady
x0, y0 = (0.3 if uinf else 0.5), 0.5


def exact(t):
  """Exact isentropic-vortex state (rho, u, v, p) at time t (radius Rc)."""
  xv, yv = x0 + uinf * t, y0 + vinf * t
  dx, dy = xc - xv, yc - yv
  r2 = (dx * dx + dy * dy) / (Rc * Rc)
  fac = beta / (2 * np.pi) * np.exp(0.5 * (1 - r2))
  u = uinf - (dy / Rc) * fac
  v = vinf + (dx / Rc) * fac
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

# final L2 error vs exact over the whole domain (the far field is uniform = exact)
re, ue, ve, pe = exact(t)
l2_rho = np.sqrt(np.sum(vol * (rho.cell - re) ** 2) / np.sum(vol))
if RANK == 0:
  kind = "steady (overlaps exact)" if uinf == 0 else f"advected by uinf*t={uinf*t:.3f}"
  print(f"isentropic vortex (Rc={Rc}, UINF={uinf}) at t={t:.3f} -- {kind}")
  print(f"  L2(rho) vs exact = {l2_rho:.3e}")
  if output_every:
    print(f"  wrote {miter + 1} VTK snapshots to ./vtk_results/ "
          f"(fields: rho/rho_exact/rho_err, P/P_exact, u/u_exact)")
