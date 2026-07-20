#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-layer lock-exchange — dynamic validation of the multilayer solver.

A dense bottom layer (left) and a lighter one (right) are separated by a gate at
x0. When released, the dense fluid runs right along the bed as a gravity current
while the light fluid runs left along the surface. The gravity-current nose speed
is compared to the Boussinesq lock-exchange estimate u_f ~ 0.5*sqrt(g' H), with
g' = g (rho1 - rho2)/rho2 and H the total depth.

Writes VTK frames (h1, h2, s1, s2) for ParaView.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

# ---- setup -----------------------------------------------------------------
Lx, Ly = 1.0, 0.1
mesh = Mesh.rectangle(bounds=((0., Lx), (0., Ly)), n=(200, 8), cell_type="triangle")
domain = mesh.domain
xc = np.asarray(domain.cells.center)[:, 0]

rho1, rho2 = 1030., 1000.          # dense bottom, light top
H = 0.10                            # total depth (flat free surface)
x0 = 0.5                            # gate position
hmin, hmax = 0.01, H - 0.01        # 90%-depth lock (avoid zero-thickness layers)
grav = 9.81
gp = grav * (rho1 - rho2) / rho2   # reduced gravity
uf_theory = 0.5 * np.sqrt(gp * H)

# left: dense-heavy (thick bottom), right: light-heavy (thin bottom)
h1_0 = np.where(xc < x0, hmax, hmin)
h2_0 = H - h1_0

layers = [
  {'h':  mesh.field("h1",  init=h1_0, bc=NEU),
   'hu': mesh.field("hu1", init=0.0,  bc=NEU),
   'hv': mesh.field("hv1", init=0.0,  bc=NEU),
   's':  mesh.field("s1",  init=np.where(xc < x0, 1.0, 0.0), bc=NEU)},   # dye marks the dense water
  {'h':  mesh.field("h2",  init=h2_0, bc=NEU),
   'hu': mesh.field("hu2", init=0.0,  bc=NEU),
   'hv': mesh.field("hv2", init=0.0,  bc=NEU),
   's':  mesh.field("s2",  init=0.0,  bc=NEU)},
]

S = MultilayerSWSolver(layers, rho=[rho1, rho2], grav=grav, cfl=0.8, order=1)

if RANK == 0:
  print(f"[lock] g'={gp:.4f}  H={H}  u_f(theory)=0.5*sqrt(g'H)={uf_theory:.4f} m/s")

# ---- time loop -------------------------------------------------------------
Tfinal = 3.0
h_thresh = hmin + 0.2 * (hmax - hmin)   # nose detection threshold on h1
time, niter, miter = 0.0, 0, 0

def nose_position():
  h1 = np.asarray(layers[0]['h'].cell)
  mask = (h1 > h_thresh) & (xc > x0)
  loc = np.max(xc[mask]) if np.any(mask) else x0
  return COMM.allreduce(float(loc), op=MPI.MAX)

while time < Tfinal:
  dt = S.stepper()
  time += dt
  S.compute_fluxes()
  S.compute_new_val()
  niter += 1
  if niter % 100 == 0:
    domain.save_on_cell_multi(["h1", "h2", "s1", "s2"],
                              [layers[0]['h'].cell, layers[1]['h'].cell,
                               layers[0]['s'].cell, layers[1]['s'].cell],
                              dt, time, niter, miter)
    miter += 1

xf = nose_position()
uf_meas = (xf - x0) / time

if RANK == 0:
  print(f"[lock] iters={niter}  t={time:.3f}")
  print(f"[lock] nose x_f={xf:.4f}  ->  u_f(measured)={uf_meas:.4f} m/s")
  print(f"[lock] u_f measured / theory = {uf_meas / uf_theory:.3f}")
  print(f"[lock] (order-1 SRNH is diffusive + 90%-depth lock: expect ~0.6-0.9)")
