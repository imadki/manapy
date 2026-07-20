#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense brine plume plunging down a slope with the ROBUST HLLC flux (v3).

The SRNH flux breaks at the sonic point (Froude=1): it gets the wave speeds from
the trigonometric solution of the characteristic cubic, needing arccos(R/sqrt(-Q^3)),
whose argument reaches exactly 1 at Fr=1 -> floating-point roundoff -> NaN. A dense
plume plunging down a slope crosses Fr=1, so SRNH blows up instantly here.

HLLC (`scheme='hllc'`) needs only wave-speed estimates u +/- sqrt(g h) -- no arccos,
no eigen-cubic -- so it stays finite through Fr=1. Well-balancing is by Audusse's
hydrostatic reconstruction; the baroclinic coupling still enters via the effective
bed. Entrainment (v2) then dilutes the current as it accelerates.

Writes the bottom-layer concentration c1 (the dilution map) to VTK for ParaView.
Run with:  python3 -u brine_plume_hllc2d.py
"""
import sys
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

L, W = 1.5, 0.10
ETA, SLOPE = 0.15, 0.06            # sea level, 6% seabed slope (SRNH blows up here)
RHO1, RHO2 = 1035., 1000.          # brine, seawater
GRAV = 9.81
H_FILM, H_POOL, X_LOCK = 0.004, 0.05, 0.25
MANN = 0.01
TFIN = 3.0

def log(m):
  if RANK == 0:
    print(m, flush=True)

log("[hllc] building + JIT compiling (~a few s) ...")
mesh = Mesh.rectangle(bounds=((0., L), (0., W)), n=(120, 5), cell_type="triangle")
domain = mesh.domain
xc = np.asarray(domain.cells.center)[:, 0]
vol = np.asarray(domain.cells.volume)

Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)
h1_0 = H_FILM + (H_POOL - H_FILM) * 0.5 * (1.0 - np.tanh((xc - X_LOCK) / 0.05))
D = ETA + SLOPE * xc
layers = [
  {'h': mesh.field("h1", init=h1_0, bc=NEU), 'hu': mesh.field("hu1", init=0.0, bc=NEU),
   'hv': mesh.field("hv1", init=0.0, bc=NEU), 's': mesh.field("s1", init=h1_0, bc=NEU)},
  {'h': mesh.field("h2", init=D - h1_0, bc=NEU), 'hu': mesh.field("hu2", init=0.0, bc=NEU),
   'hv': mesh.field("hv2", init=0.0, bc=NEU), 's': mesh.field("s2", init=0.0, bc=NEU)},
]
# scheme='hllc' is the whole point: robust through the transcritical plunge.
S = MultilayerSWSolver(layers, rho=[RHO1, RHO2], Z=Z, grav=GRAV, cfl=0.45, order=1,
                       scheme="hllc", entrain=True, E0=0.075, Mann=MANN)
gp = GRAV * (RHO1 - RHO2) / RHO2
log(f"[hllc] plunging brine plume, slope={SLOPE*100:.0f}%, g'={gp:.3f}  (SRNH: instant NaN)")

time, niter, miter = 0.0, 0, 0
while time < TFIN:
  dt = S.stepper()
  time += dt
  S.compute_fluxes()
  S.compute_new_val()
  niter += 1
  h1 = np.asarray(layers[0]['h'].cell)
  if not np.all(np.isfinite(h1)):
    log(f"[hllc] unexpected blow-up at t={time:.3f}")
    break
  if niter % 300 == 0:
    s1 = np.asarray(layers[0]['s'].cell)
    m = h1 > 2 * H_FILM
    u1 = np.abs(np.asarray(layers[0]['hu'].cell)) / np.maximum(h1, 1e-9)
    Fr = u1.max() / np.sqrt(gp * max(h1.max(), 1e-9))
    conc = float(np.sum(s1[m] * vol[m]) / max(np.sum(h1[m] * vol[m]), 1e-12))
    log(f"  t={time:4.2f} it={niter:5d} max|u|={u1.max():.3f} Fr~{Fr:.2f} "
        f"nose={np.max(xc[m]):.2f} mean_conc={conc:.4f}")
    c1 = s1 / np.maximum(h1, 1e-12)
    domain.save_on_cell_multi(["h1", "c1", "hu1"],
                              [layers[0]['h'].cell, c1, layers[0]['hu'].cell],
                              dt, time, niter, miter)
    miter += 1

log(f"[hllc] done: t={time:.2f}, iters={niter} (stable through the transcritical plunge)")
