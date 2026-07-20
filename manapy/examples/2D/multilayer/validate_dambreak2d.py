#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXACT-solution validation of the multilayer solver at N=1.

With a single layer the multilayer system reduces exactly to the single-layer
shallow-water equations (the effective bed is just the real bed, no coupling).
The wet/wet dam-break then has the classical **Stoker exact solution** (rarefaction
+ shock), so we get a genuine full-profile reference — not just a front speed.

We report the L2/Linf error against the exact profile and the L2 convergence
order under mesh refinement. VTK frames (h, h_exact) are written for ParaView.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

GRAV = 9.81
HL, HR = 0.020, 0.005              # left/right depths (wet/wet)
X0 = 0.5
TFIN = 0.06                        # short enough that no wave reaches a wall


def stoker_star(hL, hR, g):
  """Solve the dam-break star state (h*, u*) by bisection on h*."""
  cL = np.sqrt(g * hL)
  def f(hs):
    cs = np.sqrt(g * hs)
    u_raref = 2.0 * (cL - cs)                                   # left rarefaction
    u_shock = (hs - hR) * np.sqrt(0.5 * g * (hs + hR) / (hs * hR))  # right shock
    return u_raref - u_shock
  lo, hi = hR, hL
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if f(lo) * f(mid) <= 0.0:
      hi = mid
    else:
      lo = mid
  hs = 0.5 * (lo + hi)
  us = 2.0 * (cL - np.sqrt(g * hs))
  return hs, us


def stoker_profile(x, t, hL, hR, g, x0=X0):
  cL = np.sqrt(g * hL)
  hs, us = stoker_star(hL, hR, g)
  cs = np.sqrt(g * hs)
  s = hs * us / (hs - hR)                    # shock speed
  xi = (x - x0) / t
  h = np.empty_like(x)
  for i in range(len(x)):
    z = xi[i]
    if z < -cL:
      h[i] = hL
    elif z < us - cs:                        # rarefaction fan
      c = (2.0 * cL - z) / 3.0
      h[i] = c * c / g
    elif z < s:                              # star region
      h[i] = hs
    else:
      h[i] = hR
  return h


def run(nx):
  mesh = Mesh.rectangle(bounds=((0., 1.), (0., 0.05)), n=(nx, 4), cell_type="triangle")
  domain = mesh.domain
  xc = np.asarray(domain.cells.center)[:, 0]

  h0 = np.where(xc < X0, HL, HR)
  layers = [{
    'h':  mesh.field("h",  init=h0,  bc=NEU),
    'hu': mesh.field("hu", init=0.0, bc=NEU),
    'hv': mesh.field("hv", init=0.0, bc=NEU),
    's':  mesh.field("s",  init=0.0, bc=NEU),
  }]
  S = MultilayerSWSolver(layers, rho=[1000.], grav=GRAV, cfl=0.8, order=1)

  time = 0.0
  while time < TFIN:
    dt = S.stepper()
    if time + dt > TFIN:
      dt = TFIN - time
      S.dt = dt
    time += dt
    S.compute_fluxes()
    S.compute_new_val()

  h_num = np.asarray(layers[0]['h'].cell)
  h_ex = stoker_profile(xc, time, HL, HR, GRAV)
  vol = np.asarray(domain.cells.volume)
  err = h_num - h_ex
  num = COMM.allreduce(float(np.sum(vol * err * err)), op=MPI.SUM)
  den = COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
  l2 = np.sqrt(num / den)
  linf = COMM.allreduce(float(np.max(np.abs(err))), op=MPI.MAX)

  # VTK frame for ParaView (numerical + exact)
  domain.save_on_cell_multi(["h", "h_exact"], [h_num, h_ex], dt, time, 1, 0)
  return l2, linf


if RANK == 0:
  print(f"[dambreak] exact Stoker solution, hL={HL} hR={HR} g={GRAV}, T={TFIN}")

prev = None
for nx in (100, 200, 400):
  l2, linf = run(nx)
  if RANK == 0:
    order = "" if prev is None else f"  order={np.log(prev / l2) / np.log(2.0):.2f}"
    print(f"[dambreak] nx={nx:4d}   L2={l2:.3e}   Linf={linf:.3e}{order}")
  prev = l2
