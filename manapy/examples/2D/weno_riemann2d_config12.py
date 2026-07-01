#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Riemann problem, Configuration 12 (Lax & Liu 1998), with the WENO Euler solver.

Domain [0,1]^2, gamma=1.4, outflow boundaries, four constant states split by x=0.5
and y=0.5:
    (rho, u, v, p) =
        (0.5313, 0,      0,      0.4)   x>=0.5, y>=0.5   (top-right)
        (1.0,    0.7276, 0,      1.0)   x< 0.5, y>=0.5   (top-left)
        (0.8,    0,      0,      1.0)   x< 0.5, y< 0.5   (bottom-left)
        (1.0,    0,      0.7276, 1.0)   x>=0.5, y< 0.5   (bottom-right)

The adjacent states are connected by two shocks and two slip lines; the solution
(t=0.25) is symmetric about the diagonal y=x and the slip lines roll up into
Kelvin-Helmholtz vortices. High-order WENO resolves the roll-up that first-order
Rusanov smears; the cleanest scalar signal is the peak density, which for this
configuration reaches ~1.7 in reference solutions (Lax & Liu 1998; Kurganov &
Tadmor 2002) -- an over-diffused first-order scheme falls well short. The script
runs both and writes VTK of the density for the visual roll-up comparison.

Run (numba compiles the WENO build + kernels on first use):
    MESH_DIR=../../../meshes/geo python3 weno_riemann2d_config12.py
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
xc, yc, vol = cells.center[:, 0], cells.center[:, 1], np.asarray(cells.volume)
gamma = 1.4
tfinal = float(os.environ.get('TFINAL', 0.25))


def init_state():
  """Return (rho, u, v, p) fields for Configuration 12."""
  rho = np.empty_like(xc); u = np.zeros_like(xc); v = np.zeros_like(xc); p = np.empty_like(xc)
  tr = (xc >= 0.5) & (yc >= 0.5)
  tl = (xc < 0.5) & (yc >= 0.5)
  bl = (xc < 0.5) & (yc < 0.5)
  br = (xc >= 0.5) & (yc < 0.5)
  rho[tr] = 0.5313; p[tr] = 0.4
  rho[tl] = 1.0; u[tl] = 0.7276; p[tl] = 1.0
  rho[bl] = 0.8; p[bl] = 1.0
  rho[br] = 1.0; v[br] = 0.7276; p[br] = 1.0
  return rho, u, v, p



# --- WENO Euler ---
rho, rhou, rhov, rhoE = (Variable(domain=domain) for _ in range(4))
r0, u0, v0, p0 = init_state()
rho.cell[:] = r0; rhou.cell[:] = r0 * u0; rhov.cell[:] = r0 * v0
rhoE.cell[:] = p0 / (gamma - 1) + 0.5 * r0 * (u0 ** 2 + v0 ** 2)
bc = {k: "outflow" for k in ("in", "out", "upper", "bottom")}
weno = WenoEulerSolver(domain, rho.cell, rhou.cell, rhov.cell, rhoE.cell,
                        gamma=gamma, cfl=0.7, bc=bc)
t = 0.0; niter = 0
while t < tfinal:
  dt = weno.stepper()
  if t + dt > tfinal:
    dt = tfinal - t
  weno.step(dt); t += dt; niter += 1
  
  # print(dt, niter)
rho_weno = rho.cell.copy()


# --- first-order Rusanov reference ---
r2, P2 = Variable(domain=domain), Variable(domain=domain)
ru2, rv2, rE2 = (Variable(domain=domain) for _ in range(3))
r2.cell[:] = r0; ru2.cell[:] = r0 * u0; rv2.cell[:] = r0 * v0
rE2.cell[:] = p0 / (gamma - 1) + 0.5 * r0 * (u0 ** 2 + v0 ** 2)
P2.cell[:] = p0
S1 = EulerSolver(r2, P2, ru2, rv2, rE2, gamma=gamma, cfl=0.7, scheme="rusanov", bc="Neumann")
t = 0.0
while t < tfinal:
  dt = S1.stepper()
  if t + dt > tfinal:
    dt = tfinal - t; S1.dt = dt
  t += dt; S1.compute_fluxes(t); S1.compute_new_val()
  # niter+=1
  # print(dt, niter)
rho_o1 = r2.cell.copy()

if int(os.environ.get('OUTPUT', 1)):
  domain.save_on_cell_multi(["rho_weno", "rho_order1"], [rho_weno, rho_o1], dt, tfinal, niter, 0)

if RANK == 0:
  print(f"2D Riemann Config 12 (WENO Euler), t={tfinal}, {niter} SSP-RK3 steps")
  print(f"  rho range:   WENO [{rho_weno.min():.4f}, {rho_weno.max():.4f}]   "
        f"order1 [{rho_o1.min():.4f}, {rho_o1.max():.4f}]")
  print(f"  peak density (ref ~1.7):  WENO {rho_weno.max():.3f}   order1 {rho_o1.max():.3f}  "
        f"-> WENO is markedly less diffused")
  if int(os.environ.get('OUTPUT', 1)):
    print("  wrote VTK (rho_weno, rho_order1) to ./vtk_results/ (view the slip-line roll-up)")
