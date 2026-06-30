#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-boundary BC dispatch demo: a channel with mixed boundary conditions.

EulerSolver accepts a per-boundary map {boundary_name: type} so that different
boundaries get different treatments in a single run -- the enabler for realistic
configurations (jets, shear layers, flames in a channel). Supported types:
  'neumann'       zero-gradient,
  'slipwall'      reflect the wall-normal velocity (inviscid wall),
  'nonreflecting' characteristic far-field (Riemann invariants).

Here: slip walls on the lateral boundaries (upper/bottom) and a non-reflecting
inlet/outlet (in/out). A uniform stream stays perfectly steady, and an acoustic
disturbance leaves axially instead of being trapped.

Run:
    MESH_DIR=../../../meshes/geo python3 mixed_bc_channel2d.py
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

gamma = 1.4
rho0, u0, p0 = 1.0, 0.5, 1.0

rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)

# uniform stream + a small Gaussian pressure pulse at the centre
r2 = (xc - xc.mean()) ** 2 + (yc - yc.mean()) ** 2
P.cell[:] = p0 + 0.1 * np.exp(-r2 / 0.05 ** 2)
rho.cell[:] = rho0 * (P.cell / p0) ** (1.0 / gamma)
rhou.cell[:] = rho.cell * u0
rhov.cell[:] = 0.0
rhoE.cell[:] = P.cell / (gamma - 1.0) + 0.5 * rho.cell * u0 ** 2

bc_map = {"in": "nonreflecting", "out": "nonreflecting",
          "upper": "slipwall", "bottom": "slipwall"}
solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                     scheme="rusanov", bc=bc_map,
                     rho_inf=rho0, u_inf=u0, v_inf=0.0, p_inf=p0)

e0 = float(np.sum(vol * (P.cell - p0) ** 2))
tfinal = float(os.environ.get('TFINAL', 1.5))
t = 0.0
niter = 0
while t < tfinal:
  dt = solver.stepper()
  if t + dt > tfinal:
    dt = tfinal - t; solver.dt = dt
  t += dt
  solver.compute_fluxes(t)
  solver.compute_new_val()
  niter += 1

ef = float(np.sum(vol * (P.cell - p0) ** 2))
if RANK == 0:
  print(f"channel: slip walls (upper/bottom) + non-reflecting (in/out)")
  print(f"iters={niter} t={t:.3f}")
  print(f"  acoustic residual energy {e0:.3e} -> {ef:.3e}  (pulse radiated out axially)")
  u = rhou.cell / rho.cell
  print(f"  mean axial velocity stays {u.mean():.4f} (init {u0})")
