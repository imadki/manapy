#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WENO linear advection demo: high-order, non-oscillatory transport.

Advects a profile with the unstructured WENO reconstruction (Tsoutsanis JCP 2023,
weno.py) and an upwind flux, integrated in time with SSP-RK3 (the standard pairing
-- forward Euler is unstable with a high-order spatial scheme). Compared against a
first-order upwind scheme:
  * a square wave stays sharp and essentially non-oscillatory;
  * a Gaussian keeps its peak (~0.95) where first order smears it to ~0.58.

The mesh-dependent WENO data (stencils, pseudo-inverses, oscillation matrices) is
built once; the per-stage reconstruction and flux run compiled (numba) kernels.

VTK snapshots (WENO and first-order side by side) go to ./vtk_results/.

Run:
    MESH_DIR=../../../meshes/geo python3 weno_advection2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.weno import WenoReconstruction

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), 'carre.msh')
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc = cells.center[:, 0]
vol = np.asarray(cells.volume)
h = np.sqrt(vol.mean())
cellid = np.asarray(domain.faces.cellid)

weno = WenoReconstruction(domain, order=2)
ax, ay = 1.0, 0.0                                  # advection velocity (+x)
K = weno.K


def L(u, use_weno):
  """du/dt operator: WENO (or first-order) upwind advection residual / volume."""
  ug = u[cellid[:, 0]]                             # zero-gradient ghost
  coeffs = weno.weno_reconstruct(u) if use_weno else np.zeros((domain.nbcells, K))
  return weno.advect_residual(u, ug, coeffs, ax, ay) / vol


def ssprk3(u, use_weno, dt):
  u1 = u + dt * L(u, use_weno)
  u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1, use_weno))
  return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L(u2, use_weno))


# initial field: a square wave plus a Gaussian, both transported in +x
u0 = np.where((xc > 0.15) & (xc < 0.35), 1.0, 0.0) \
    + np.exp(-((xc - 0.55) / 0.05) ** 2)

tfinal = float(os.environ.get('TFINAL', 0.3))
output_every = int(os.environ.get('OUTPUT_EVERY', 20))
dt = 0.3 * h / abs(ax)

u_w = u0.copy()
u_1 = u0.copy()
t = 0.0
niter = 0
miter = 0


def save(t, dt, niter, miter):
  domain.save_on_cell_multi(["u_weno", "u_order1", "u_init"],
                            [u_w, u_1, u0], dt, t, niter, miter)


if output_every:
  save(0.0, 0.0, 0, miter); miter += 1
while t < tfinal:
  if t + dt > tfinal:
    dt = tfinal - t
  u_w = ssprk3(u_w, True, dt)
  u_1 = ssprk3(u_1, False, dt)
  t += dt
  niter += 1
  if output_every and niter % output_every == 0:
    save(t, dt, niter, miter); miter += 1
if output_every:
  save(t, dt, niter, miter)

if RANK == 0:
  ov_w = max(u_w.max() - 1.0, -u_w.min(), 0.0)
  ov_1 = max(u_1.max() - 1.0, -u_1.min(), 0.0)
  # Gaussian peak (the bump region)
  bump = xc > 0.45
  print(f"WENO advection (SSP-RK3) to t={t:.3f}, velocity ({ax},{ay})")
  print(f"  square wave overshoot:  WENO {ov_w:.3f}   order-1 {ov_1:.3f}")
  print(f"  Gaussian peak:          WENO {u_w[bump].max():.3f}   order-1 {u_1[bump].max():.3f}  (init 1.000)")
  if output_every:
    print(f"  wrote {miter + 1} VTK snapshots to ./vtk_results/ (u_weno, u_order1, u_init)")
