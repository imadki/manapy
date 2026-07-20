#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, verbose brine-plume run — for interactive checking (run with: python3 -u).

Small mesh, single run, prints a flushed progress line every 200 steps so you can
see it move. ~30 s of silence first = numba JIT compiling the kernels (normal).
"""
import sys
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

L, W = 2.0, 0.10
ETA, SLOPE = 0.15, 0.02
RHO1, RHO2 = 1018., 1000.
GRAV, H_FILM, H_POOL, X_LOCK, MANN = 9.81, 0.004, 0.05, 0.30, 0.05
TFIN = 6.0

def log(msg):
  if RANK == 0:
    print(msg, flush=True)               # flush => visible immediately (also use python3 -u)

log("[quick] building mesh + JIT-compiling kernels (~30 s of silence is normal) ...")
mesh = Mesh.rectangle(bounds=((0., L), (0., W)), n=(120, 6), cell_type="triangle")
domain = mesh.domain
xc = np.asarray(domain.cells.center)[:, 0]
vol = np.asarray(domain.cells.volume)

Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)
h1_0 = H_FILM + (H_POOL - H_FILM) * 0.5 * (1.0 - np.tanh((xc - X_LOCK) / 0.10))
D = ETA + SLOPE * xc
layers = [
  {'h': mesh.field("h1", init=h1_0, bc=NEU), 'hu': mesh.field("hu1", init=0.0, bc=NEU),
   'hv': mesh.field("hv1", init=0.0, bc=NEU), 's': mesh.field("s1", init=h1_0, bc=NEU)},
  {'h': mesh.field("h2", init=D - h1_0, bc=NEU), 'hu': mesh.field("hu2", init=0.0, bc=NEU),
   'hv': mesh.field("hv2", init=0.0, bc=NEU), 's': mesh.field("s2", init=0.0, bc=NEU)},
]
S = MultilayerSWSolver(layers, rho=[RHO1, RHO2], Z=Z, grav=GRAV, cfl=0.5, order=1,
                       entrain=True, E0=0.075, Mann=MANN)
log("[quick] compiled. running ...")

time, niter, miter = 0.0, 0, 0
while time < TFIN:
  dt = S.stepper()
  time += dt
  S.compute_fluxes()
  S.compute_new_val()
  niter += 1
  h1 = np.asarray(layers[0]['h'].cell)
  if not np.all(np.isfinite(h1)):
    log(f"[quick] BLOW-UP at t={time:.3f} (SRNH transcritical) — lower slope or raise Mann")
    break
  if niter % 200 == 0:
    s1 = np.asarray(layers[0]['s'].cell)
    m = h1 > 2 * H_FILM
    u1 = np.abs(np.asarray(layers[0]['hu'].cell)) / np.maximum(h1, 1e-9)
    conc = float(np.sum(s1[m] * vol[m]) / max(np.sum(h1[m] * vol[m]), 1e-12))
    log(f"  t={time:5.2f}  it={niter:5d}  dt={dt:.1e}  nose_x={np.max(xc[m]):.2f}  "
        f"min_h1={h1.min():.4f}  max|u1|={u1.max():.3f}  mean_conc={conc:.3f}")
  if niter % 300 == 0:
    c1 = np.asarray(layers[0]['s'].cell) / np.maximum(h1, 1e-12)
    domain.save_on_cell_multi(["h1", "c1"], [layers[0]['h'].cell, c1], dt, time, niter, miter)
    miter += 1

log(f"[quick] done: t={time:.2f}, iters={niter}")
