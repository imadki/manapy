#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Well-balanced test (C-property) for the multilayer solver.

Two layers at rest with a flat interface: a dense bottom layer under a lighter
top layer, no forcing. A well-balanced coupling must keep the fluid at rest:
max|u| must stay at ~machine level, NOT grow. This validates that the effective-
bed baroclinic source exactly balances the layer flux at the stratified rest
state before we trust any dynamic run.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

mesh = Mesh.rectangle(bounds=((0., 1.), (0., 0.1)), n=(80, 8), cell_type="triangle")

rho = [1030., 1000.]          # bottom (dense) -> top (light)
h1, h2 = 0.05, 0.05           # flat interface, total depth 0.10

layers = []
for k, hk in enumerate((h1, h2)):
  lay = {
    'h':  mesh.field(f"h{k+1}",  init=hk,          bc=NEU),
    'hu': mesh.field(f"hu{k+1}", init=0.0,         bc=NEU),
    'hv': mesh.field(f"hv{k+1}", init=0.0,         bc=NEU),
    's':  mesh.field(f"s{k+1}",  init=(hk if k == 0 else 0.0), bc=NEU),
  }
  layers.append(lay)

S = MultilayerSWSolver(layers, rho=rho, grav=9.81, cfl=0.8, order=1)

if RANK == 0:
  print("[rest] stratified rest state, checking C-property ...")

time, niter = 0.0, 0
Tfinal = 2.0
while time < Tfinal:
  dt = S.stepper()
  time += dt
  S.compute_fluxes()
  S.compute_new_val()
  niter += 1

# diagnostics: max speed and interface drift
umax = 0.0
for lay in layers:
  spd = np.max(np.abs(np.asarray(lay['hu'].cell)) + np.abs(np.asarray(lay['hv'].cell)))
  umax = max(umax, float(spd))
umax = COMM.allreduce(umax, op=MPI.MAX)

h1_dev = float(np.max(np.abs(np.asarray(layers[0]['h'].cell) - h1)))
h1_dev = COMM.allreduce(h1_dev, op=MPI.MAX)

if RANK == 0:
  print(f"[rest] iters={niter}  t={time:.3f}")
  print(f"[rest] max|h1*u| over both layers = {umax:.3e}   (want ~1e-12)")
  print(f"[rest] max|h1 - h1_0|            = {h1_dev:.3e}   (want ~1e-12)")
  ok = umax < 1e-9
  print("[rest] C-PROPERTY:", "PASS" if ok else "FAIL")
