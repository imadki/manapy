#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense brine plume plunging down a slope — v2 demo + entrainment validation.

A pool of dense brine is released at the top of a sloping seabed. It plunges
downslope as a gravity current (driven by the effective-bed baroclinic source),
accelerates to a low Richardson number, and entrains ambient seawater — the
physical regime where entrainment matters (unlike the balanced flat lock-exchange,
which sits at Ri~4 and barely mixes).

Run WITH and WITHOUT entrainment to isolate its effect:
  * with entrainment the current DILUTES (mean concentration drops) and THICKENS;
  * the C-property still holds (ambient stays at rest until the plume arrives).

Writes the bottom-layer concentration c1 (the dilution/hypersalinity map) to VTK.
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

L, W = 2.0, 0.10
ETA = 0.15                 # sea level (flat free surface reference)
SLOPE = 0.02               # gentle seabed slope (keeps the current subcritical for SRNH)
RHO1, RHO2 = 1018., 1000.  # brine, seawater (moderate contrast)
GRAV = 9.81
H_FILM = 0.004             # thin residual dense film (avoids zero thickness)
H_POOL = 0.05              # dense pool depth at the top of the slope
X_LOCK = 0.30              # pool extends over x < X_LOCK
MANN = 0.05                # strong bottom drag -> quasi-steady, subcritical current
TFIN = 12.0


def build(mesh):
  domain = mesh.domain
  xc = np.asarray(domain.cells.center)[:, 0]
  Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)          # sloping bed
  # smoothed dense pool at the top of the slope (wide transition -> gentler front)
  h1_0 = H_FILM + (H_POOL - H_FILM) * 0.5 * (1.0 - np.tanh((xc - X_LOCK) / 0.10))
  D = ETA - (-SLOPE * xc)                                        # total depth = eta - Z
  h2_0 = D - h1_0
  layers = [
    {'h':  mesh.field("h1",  init=h1_0, bc=NEU),
     'hu': mesh.field("hu1", init=0.0,  bc=NEU),
     'hv': mesh.field("hv1", init=0.0,  bc=NEU),
     's':  mesh.field("s1",  init=h1_0, bc=NEU)},               # concentration 1 in the brine
    {'h':  mesh.field("h2",  init=h2_0, bc=NEU),
     'hu': mesh.field("hu2", init=0.0,  bc=NEU),
     'hv': mesh.field("hv2", init=0.0,  bc=NEU),
     's':  mesh.field("s2",  init=0.0,  bc=NEU)},
  ]
  return domain, xc, Z, layers


def current_diag(layers, vol):
  """Volume-weighted mean concentration of the moving dense current + its volume."""
  h1 = np.asarray(layers[0]['h'].cell)
  s1 = np.asarray(layers[0]['s'].cell)
  mask = h1 > 2.0 * H_FILM
  sm = float(np.sum(s1[mask] * vol[mask]))
  hm = float(np.sum(h1[mask] * vol[mask]))
  sm = COMM.allreduce(sm, op=MPI.SUM)
  hm = COMM.allreduce(hm, op=MPI.SUM)
  conc = sm / hm if hm > 0 else 0.0
  vtot = COMM.allreduce(float(np.sum(h1 * vol)), op=MPI.SUM)
  return conc, vtot


def run(entrain, save=False):
  mesh = Mesh.rectangle(bounds=((0., L), (0., W)), n=(240, 8), cell_type="triangle")
  domain, xc, Z, layers = build(mesh)
  vol = np.asarray(domain.cells.volume)
  S = MultilayerSWSolver(layers, rho=[RHO1, RHO2], Z=Z, grav=GRAV, cfl=0.5, order=1,
                         entrain=entrain, E0=0.075, Mann=MANN)
  time, niter, miter = 0.0, 0, 0
  while time < TFIN:
    dt = S.stepper()
    time += dt
    S.compute_fluxes()
    S.compute_new_val()
    niter += 1
    if not np.all(np.isfinite(np.asarray(layers[0]['h'].cell))):
      if RANK == 0:
        print(f"  [!] non-finite h1 at t={time:.3f} (SRNH transcritical blow-up)")
      break
    if save and niter % 1 == 0:
      h1 = np.asarray(layers[0]['h'].cell)
      c1 = np.asarray(layers[0]['s'].cell) / np.maximum(h1, 1e-12)
      domain.save_on_cell_multi(["h1", "c1", "hu1"],
                                [layers[0]['h'].cell, c1, layers[0]['hu'].cell],
                                dt, time, niter, miter)
      miter += 1
  conc, vtot = current_diag(layers, vol)
  # nose position (furthest down-slope reach of the dense current)
  h1 = np.asarray(layers[0]['h'].cell)
  m = h1 > 2.0 * H_FILM
  xnose = COMM.allreduce(float(np.max(xc[m]) if np.any(m) else 0.0), op=MPI.MAX)
  return conc, vtot, xnose


if RANK == 0:
  gp = GRAV * (RHO1 - RHO2) / RHO2
  print(f"[brine] dense plume on {SLOPE*100:.0f}% slope, g'={gp:.3f}, T={TFIN}s")

c_no, v_no, x_no = run(entrain=False)
c_en, v_en, x_en = run(entrain=True, save=True)

if RANK == 0:
  print(f"[brine] no-entrain : mean current conc = {c_no:.3f}   dense vol = {v_no:.4e}   nose x = {x_no:.2f}")
  print(f"[brine] entrain    : mean current conc = {c_en:.3f}   dense vol = {v_en:.4e}   nose x = {x_en:.2f}")
  if c_en > 0 and v_no > 0 and np.isfinite(c_no) and np.isfinite(c_en):
    print(f"[brine] dilution (conc drop) from entrainment = {c_no / c_en:.2f}x   "
          f"({c_en:.3f} vs {c_no:.3f})")
    print(f"[brine] thickening from entrainment           = {v_en / v_no:.2f}x")
    print("[brine] DILUTION VISIBLE:", "PASS" if c_en < 0.95 * c_no else "weak (needs lower Ri)")
  else:
    print("[brine] run did not complete cleanly (see blow-up note above)")
