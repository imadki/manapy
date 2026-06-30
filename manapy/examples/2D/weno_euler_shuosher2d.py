#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shu-Osher problem with the WENO Euler solver (the classic WENO stress test).

A Mach-3 shock runs into a field with a sinusoidal density perturbation; behind
the shock, high-frequency acoustic/entropy structures are generated. A good
high-order non-oscillatory scheme must capture the shock AND resolve the smeared,
entrained post-shock waves -- exactly where a first-order scheme fails.

This is a 1-D problem run on a 2-D unstructured mesh (the field is uniform in y);
it needs good x-resolution, which the bundled square mesh provides. The example
runs WENO and first-order Rusanov and reports the surviving post-shock oscillation
amplitude (WENO keeps it; first order diffuses it away). VTK of both goes to
./vtk_results/.

Run:
    MESH_DIR=../../../meshes/geo python3 weno_euler_shuosher2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.weno_euler import WenoEulerSolver
from manapy.solvers.euler.system import EulerSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), 'carre.msh')
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc, yc = cells.center[:, 0], cells.center[:, 1]
gamma = 1.4

# Shu-Osher scaled to [0,1]: shock at x=0.1, post-shock state to the left,
# 1 + 0.2 sin(50 x) density to the right.
xs = 0.1
rhoL, uL, pL = 3.857143, 2.629369, 10.33333


def initialize(rho, rhou, rhov, rhoE):
  left = xc < xs
  r = np.where(left, rhoL, 1.0 + 0.2 * np.sin(50.0 * xc))
  u = np.where(left, uL, 0.0)
  p = np.where(left, pL, 1.0)
  rho[:] = r; rhou[:] = r * u; rhov[:] = 0.0
  rhoE[:] = p / (gamma - 1) + 0.5 * r * u * u


tfinal = float(os.environ.get('TFINAL', 0.18))

# --- WENO Euler ---
rho = Variable(domain=domain); rhou = Variable(domain=domain)
rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
initialize(rho.cell, rhou.cell, rhov.cell, rhoE.cell)
bc = {"in": {"rho": rhoL, "u": uL, "v": 0.0, "p": pL},
      "out": "outflow", "upper": "outflow", "bottom": "outflow"}
weno = WenoEulerSolver(domain, rho.cell, rhou.cell, rhov.cell, rhoE.cell,
                       gamma=gamma, cfl=0.3, bc=bc)
output_every = int(os.environ.get('OUTPUT_EVERY', 30))
t = 0.0; niter = 0; miter = 0
while t < tfinal:
  dt = weno.stepper()
  if t + dt > tfinal:
    dt = tfinal - t
  weno.step(dt); t += dt; niter += 1
rho_weno = rho.cell.copy()

# --- first-order Rusanov reference ---
r2 = Variable(domain=domain); P2 = Variable(domain=domain)
ru2 = Variable(domain=domain); rv2 = Variable(domain=domain); rE2 = Variable(domain=domain)
initialize(r2.cell, ru2.cell, rv2.cell, rE2.cell)
P2.cell[:] = (gamma - 1) * (rE2.cell - 0.5 * ru2.cell ** 2 / r2.cell)
S1 = EulerSolver(r2, P2, ru2, rv2, rE2, gamma=gamma, cfl=0.3, scheme="rusanov",
                 bc={"in": "nonreflecting", "out": "nonreflecting",
                     "upper": "slipwall", "bottom": "slipwall"},
                 rho_inf=rhoL, u_inf=uL, v_inf=0.0, p_inf=pL)
t = 0.0
while t < tfinal:
  dt = S1.stepper()
  if t + dt > tfinal:
    dt = tfinal - t; S1.dt = dt
  t += dt; S1.compute_fluxes(t); S1.compute_new_val()
rho_o1 = r2.cell.copy()

if output_every:
  domain.save_on_cell_multi(["rho_weno", "rho_order1"], [rho_weno, rho_o1], dt, t, niter, 0)

if RANK == 0:
  band = np.abs(yc - 0.5) < 0.05
  post = band & (xc > 0.35) & (xc < 0.62)        # entrained post-shock waves

  def amp(r):
    rp = r[post]; return rp.max() - rp.min()
  print(f"Shu-Osher (WENO Euler, t={t:.3f}, {niter} steps)")
  print(f"  post-shock oscillation amplitude:  WENO {amp(rho_weno):.3f}   order-1 {amp(rho_o1):.3f}")
  print(f"  -> WENO keeps {amp(rho_weno) / max(amp(rho_o1), 1e-9):.1f}x more of the entrained structure")
  if output_every:
    print("  wrote VTK (rho_weno, rho_order1) to ./vtk_results/")
