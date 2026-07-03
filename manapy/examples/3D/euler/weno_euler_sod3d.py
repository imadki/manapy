#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D WENO Euler solver on the Sod shock tube (3D analogue of weno_euler_sod2d.py).

Uses WenoEulerSolver in 3D: per-variable unstructured WENO reconstruction of the
five conservative variables (rho, rho*u, rho*v, rho*w, rho*E) + Rusanov flux at the
face centres + SSP-RK3. The problem is 1-D in x on the 3D box mesh (uniform in y,z).
A VTK snapshot of the density is written each output step. The WENO solution is
sharper and non-oscillatory at the shock/contact compared to first order.

Run (numba compiles the 3D WENO build + kernels on first use):
    MESH_DIR=../../../../meshes python3 weno_euler_sod3d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.weno_euler import WenoEulerSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', 'meshes')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), os.environ.get('MESH_FILE', 'hybrid3d.msh'))
domain = Domain.create_domain(mesh, 3, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc, vol = cells.center[:, 0], np.asarray(cells.volume)
gamma = 1.4
xm = 0.5 * (xc.min() + xc.max())

rho = Variable(domain=domain); rhou = Variable(domain=domain); rhov = Variable(domain=domain)
rhow = Variable(domain=domain); rhoE = Variable(domain=domain)
left = xc < xm
rho.cell[:] = np.where(left, 1.0, 0.125)
rhoE.cell[:] = np.where(left, 1.0, 0.1) / (gamma - 1)

bc = {k: "outflow" for k in ("in", "out", "upper", "bottom", "front", "back")}
solver = WenoEulerSolver(domain, rho.cell, rhou.cell, rhov.cell, rhoE.cell, rhow=rhow.cell,
                         gamma=gamma, cfl=0.3, bc=bc)

output_every = int(os.environ.get('OUTPUT_EVERY', 10))
tfinal = float(os.environ.get('TFINAL', 0.15))
t = 0.0; niter = 0; miter = 0


def save():
  global miter
  domain.save_on_cell_multi(["rho"], [rho.cell.copy()], solver.stepper(), t, niter, miter)
  miter += 1


if output_every:
  save()
while t < tfinal:
  dt = solver.stepper()
  if t + dt > tfinal:
    dt = tfinal - t
  solver.step(dt); t += dt; niter += 1
  if output_every and niter % output_every == 0:
    save()
if output_every:
  save()

P = (gamma - 1) * (rhoE.cell - 0.5 * (rhou.cell ** 2 + rhov.cell ** 2 + rhow.cell ** 2) / rho.cell)
if RANK == 0:
  print(f"WENO-Euler 3D Sod at t={t:.3f} ({niter} SSP-RK3 steps)")
  print(f"  rho range [{rho.cell.min():.4f}, {rho.cell.max():.4f}]  (positive: {bool(np.all(rho.cell > 0))})")
  print(f"  min pressure = {P.min():.4f}")
  if output_every:
    print(f"  wrote {miter} VTK snapshots to ./vtk_results/ (rho)")
