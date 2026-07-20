#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation of the 2D viscous Burgers solver against the EXACT travelling wave.

Viscous Burgers  u_t + (u^2/2)_x = nu u_xx  admits the exact travelling-wave
(stationary viscous shock) solution

    u(x,t) = (uL+uR)/2 - (uL-uR)/2 * tanh[ (uL-uR)(x - x0 - s t) / (4 nu) ],
    s = (uL+uR)/2.

We initialise with this EXACT profile (which stays 1-D in x), evolve to T, and
compare the full numerical profile with the shifted exact solution. Running two
resolutions gives an observed convergence order.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.api.models import BurgersModel

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---- problem definition ----
uL, uR = 1.0, 0.0
nu = 0.02
x0 = 0.25
T = 0.15
s = 0.5 * (uL + uR)


def exact(x, t):
  return 0.5 * (uL + uR) - 0.5 * (uL - uR) * np.tanh((uL - uR) * (x - x0 - s * t) / (4.0 * nu))


def run_one(nx, ny=8, order=2):
  mesh = Mesh.rectangle(bounds=((0.0, 1.0), (0.0, 0.2)), n=(nx, ny), 
                        cell_type="triangle")
  # exact initial profile; Dirichlet ends held at the (saturated) exact values.
  u = mesh.field(
      "u",
      init=lambda x, y, z: exact(x, 0.0),
      bc={"in": ("dirichlet", exact(0.0, 0.0)),
          "out": ("dirichlet", exact(1.0, 0.0)),
          "upper": "neumann", "bottom": "neumann"},
      limiter="vanalbada",
  )
  BurgersModel(u, mesh, nu=nu, order=order, cfl=0.4, scheme="rusanov").run(
      T, output_every=10_000_000, output_mode="cell")  # no vtk during a convergence run

  c = np.asarray(mesh.domain.cells.center)
  uc = np.asarray(u.cell)
  ue = exact(c[:, 0], T)

  # interior mask: exclude a few cells next to the x-boundaries
  m = (c[:, 0] > 0.06) & (c[:, 0] < 0.94)
  vol = np.asarray(mesh.domain.cells.volume)[m]
  err = uc[m] - ue[m]

  # volume-weighted L2 and L-infinity (reduced across ranks)
  num = COMM.allreduce(float(np.sum(vol * err * err)), op=MPI.SUM)
  den = COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
  linf = COMM.allreduce(float(np.max(np.abs(err))) if err.size else 0.0, op=MPI.MAX)
  l2 = np.sqrt(num / den)
  return l2, linf


if RANK == 0:
  print("\n=========== Burgers 2D vs EXACT travelling wave ===========")
  print(f"  uL={uL} uR={uR} nu={nu}  T={T}  s(exact)={s}")

levels = (80, 160, 320)
results = {}
for nx in levels:
  l2, linf = run_one(nx, order=2)
  results[nx] = (l2, linf)
  if RANK == 0:
    print(f"  nx={nx:4d} (h={1.0/nx:.5f})   L2 = {l2:.3e}   Linf = {linf:.3e}")

if RANK == 0:
  for a, b in zip(levels[:-1], levels[1:]):
    l2a, l2b = results[a][0], results[b][0]
    order = np.log2(l2a / l2b) if l2b > 0 else float('nan')
    print(f"  observed L2 order ({a}->{b}) : {order:.2f}")
  print("  (limiter -> 1st order at the front when under-resolved; -> ~2 as it resolves)")
  print("===========================================================\n")
