#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation of the robust HLLC flux (v3): well-balancing + exact dam-break.

(1) C-property: a stratified rest state must stay at rest with HLLC + Audusse
    hydrostatic reconstruction (max|u| ~ machine).
(2) Exact Stoker dam-break at N=1 (solver reduces to single-layer SWE): HLLC L2
    error vs the exact rarefaction+shock profile, with convergence order.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}
GRAV = 9.81


# ---- (1) C-property with HLLC ---------------------------------------------
def cproperty(order=1):
  mesh = Mesh.rectangle(bounds=((0., 1.), (0., 0.1)), n=(80, 8), cell_type="triangle")
  layers = [
    {'h': mesh.field("h1", init=0.05, bc=NEU), 'hu': mesh.field("hu1", init=0., bc=NEU),
     'hv': mesh.field("hv1", init=0., bc=NEU), 's': mesh.field("s1", init=0.05, bc=NEU)},
    {'h': mesh.field("h2", init=0.05, bc=NEU), 'hu': mesh.field("hu2", init=0., bc=NEU),
     'hv': mesh.field("hv2", init=0., bc=NEU), 's': mesh.field("s2", init=0., bc=NEU)},
  ]
  S = MultilayerSWSolver(layers, rho=[1030., 1000.], grav=GRAV, cfl=0.5, order=order, scheme="hllc")
  t = 0.0
  while t < 1.0:
    dt = S.stepper(); t += dt
    S.compute_fluxes(); S.compute_new_val()
  umax = max(float(np.max(np.abs(np.asarray(l['hu'].cell)))) for l in layers)
  return COMM.allreduce(umax, op=MPI.MAX)


# ---- (2) exact Stoker dam-break with HLLC ---------------------------------
HL, HR, X0, TFIN = 0.020, 0.005, 0.5, 0.06

def stoker_star(hL, hR, g):
  cL = np.sqrt(g * hL)
  def f(hs):
    return 2 * (cL - np.sqrt(g * hs)) - (hs - hR) * np.sqrt(0.5 * g * (hs + hR) / (hs * hR))
  lo, hi = hR, hL
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if f(lo) * f(mid) <= 0: hi = mid
    else: lo = mid
  hs = 0.5 * (lo + hi)
  return hs, 2 * (cL - np.sqrt(g * hs))

def stoker(x, t, hL, hR, g):
  cL = np.sqrt(g * hL); hs, us = stoker_star(hL, hR, g); cs = np.sqrt(g * hs)
  s = hs * us / (hs - hR); xi = (x - X0) / t; h = np.empty_like(x)
  for i in range(len(x)):
    z = xi[i]
    if z < -cL: h[i] = hL
    elif z < us - cs: c = (2 * cL - z) / 3.0; h[i] = c * c / g
    elif z < s: h[i] = hs
    else: h[i] = hR
  return h

def dambreak(nx, order=1):
  mesh = Mesh.rectangle(bounds=((0., 1.), (0., 0.05)), n=(nx, 4), cell_type="triangle")
  domain = mesh.domain
  xc = np.asarray(domain.cells.center)[:, 0]
  h0 = np.where(xc < X0, HL, HR)
  layers = [{
    'h': mesh.field("h", init=h0, bc=NEU), 'hu': mesh.field("hu", init=0., bc=NEU),
    'hv': mesh.field("hv", init=0., bc=NEU), 's': mesh.field("s", init=0., bc=NEU)}]
  S = MultilayerSWSolver(layers, rho=[1000.], grav=GRAV, cfl=0.8, order=order, scheme="hllc")
  t = 0.0
  while t < TFIN:
    dt = S.stepper()
    if t + dt > TFIN: dt = TFIN - t; S.dt = dt
    t += dt
    S.compute_fluxes(); S.compute_new_val()
  hn = np.asarray(layers[0]['h'].cell); he = stoker(xc, t, HL, HR, GRAV)
  vol = np.asarray(domain.cells.volume); err = hn - he
  num = COMM.allreduce(float(np.sum(vol * err * err)), op=MPI.SUM)
  den = COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
  return np.sqrt(num / den)


if RANK == 0:
  print("[hllc] validating well-balanced HLLC flux ...")

for order in (1, 2):
  u = cproperty(order)
  if RANK == 0:
    print(f"[hllc] (1) C-property order={order} max|hu| = {u:.2e}  -> {'PASS' if u < 1e-9 else 'FAIL'}")

for order in (1, 2):
  prev = None
  for nx in (100, 200, 400):
    l2 = dambreak(nx, order)
    if RANK == 0:
      o = "" if prev is None else f"  order={np.log(prev / l2) / np.log(2):.2f}"
      print(f"[hllc] (2) dam-break order={order} nx={nx:4d}  L2={l2:.3e}{o}")
    prev = l2
