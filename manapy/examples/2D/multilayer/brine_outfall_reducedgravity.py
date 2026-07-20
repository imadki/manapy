#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SWRO brine outfall -- REDUCED-GRAVITY single-layer far-field model (fast + clean).

For a deep, quasi-static ambient (typical coastal outfall), only the dense brine
layer matters: it is modelled as ONE active layer under an infinite passive ambient,
driven by the REDUCED gravity g' = g (rho_brine - rho_amb)/rho_amb. This is the
classical dense-gravity-current model (Ellison-Turner / Parker), i.e. formulation (1).

Two big wins over the two-layer model on this case:
  * the fast barotropic surface wave sqrt(gH)~10 m/s is GONE -- the only wave is the
    slow internal one sqrt(g'h)~0.2 m/s, so dt is ~50x larger (fast);
  * no two-layer coupling -> NO front oscillations.

Runs the plume down a shelf slope with turbulent entrainment (single-layer form:
the current entrains ambient -> thickens -> its concentration dilutes). Writes the
bottom excess-salinity map c1 to VTK.  Run with:  python3 -u <this file>
"""
from mpi4py import MPI
import numpy as np
from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

# --- plant-scale parameters -------------------------------------------------
LX, LY = 300.0, 80.0
SLOPE = 0.03
RHO_AMB, RHO_BRINE = 1027.0, 1035.0
GRAV = 9.81
GP = GRAV * (RHO_BRINE - RHO_AMB) / RHO_AMB      # REDUCED gravity (the effective "g")
H_FILM = 0.05
X_OUT, Y_OUT, R_OUT = 30.0, 40.0, 8.0
H0, S0 = 0.5, 1.0                                  # diffuser-delivered thickness / excess-salinity
MANN = 0.02
E0 = 0.075                                         # entrainment coefficient
TFIN = 400.0

mesh = Mesh.rectangle(bounds=((0., LX), (0., LY)), n=(150, 40), cell_type="triangle")
domain = mesh.domain
cc = np.asarray(domain.cells.center)
xc, yc = cc[:, 0], cc[:, 1]
vol = np.asarray(domain.cells.volume)
w_src = np.exp(-((xc - X_OUT)**2 + (yc - Y_OUT)**2) / R_OUT**2)     # smooth diffuser footprint

Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)               # seabed
# ONE active dense layer (the ambient is passive/infinite -> not modelled)
layers = [
  {'h':  mesh.field("h1", init=np.full(len(xc), H_FILM), bc=NEU), 'hu': mesh.field("hu1", init=0., bc=NEU),
   'hv': mesh.field("hv1", init=0., bc=NEU), 's': mesh.field("s1", init=0., bc=NEU)},
]
# grav = GP (reduced): single-layer SW with reduced gravity = the gravity-current model.
# HLLC (not SRNH!): the current accelerates down-slope and goes SUPERCRITICAL (Froude>1);
# SRNH's arccos blows up at Froude=1, HLLC stays robust. And in a SINGLE layer HLLC has
# no two-layer coupling -> no front oscillations. This is the clean+fast+robust combo.
S = MultilayerSWSolver(layers, rho=[RHO_BRINE], Z=Z, grav=GP, cfl=0.5, order=1,
                       scheme="hllc", Mann=MANN)

if RANK == 0:
  print(f"[rg] REDUCED-GRAVITY outfall: g'={GP:.3f} m/s2 (vs g={GRAV}), slope {SLOPE*100:.0f}%")
  print(f"[rg] wave speed ~sqrt(g'h)={np.sqrt(GP*H0):.2f} m/s (no fast barotropic wave) -> big dt")

h1c = np.asarray(layers[0]['h'].cell); s1c = np.asarray(layers[0]['s'].cell)
hu1c = np.asarray(layers[0]['hu'].cell); hv1c = np.asarray(layers[0]['hv'].cell)
time, niter, miter = 0.0, 0, 0
while time < TFIN:
  dt = S.stepper(); time += dt
  S.compute_fluxes(); S.compute_new_val()
  # smooth steady source (near-field re-supply): relax to (h0, S0*h0)
  h1c[:] += w_src * (H0 - h1c) * 0.15
  s1c[:] += w_src * (S0 * H0 - s1c) * 0.15
  # single-layer turbulent entrainment: the current entrains ambient (at c=0) -> h grows,
  # concentration c=s/h dilutes, and u=hu/h decreases (drag by still ambient).
  h = np.maximum(h1c, 1e-9)
  spd = np.sqrt((hu1c / h)**2 + (hv1c / h)**2)
  Ri = GP * h / (spd**2 + 1e-12)
  we = E0 / np.sqrt(1.0 + 718.0 * Ri**2.4) * spd            # entrainment velocity
  h1c[:] += np.minimum(we * dt, 0.1 * h)                    # (capped for positivity)
  niter += 1
  if not np.all(np.isfinite(h1c)):
    if RANK == 0: print(f"[rg] non-finite at t={time:.1f}")
    break
  if niter % 10 == 0:
    c1 = s1c / np.maximum(h1c, 1e-9)
    domain.save_on_cell_multi(["h1", "c1", "Z"], [layers[0]['h'].cell, c1, Z.cell], dt, time, niter, miter)
    miter += 1
    if RANK == 0:
      m = h1c > 2 * H_FILM
      xr = np.max(xc[m]) if np.any(m) else X_OUT
      print(f"  t={time:5.1f}s  dt={dt:4.1f}s  reach x={xr:5.0f}m  min conc={np.min((s1c/np.maximum(h1c,1e-9))[m]):.3f}")

if RANK == 0:
  c1 = s1c / np.maximum(h1c, 1e-9); m = h1c > 2 * H_FILM
  reach = float(np.max(xc[m]) - X_OUT) if np.any(m) else 0.0
  cmin = float(np.min(c1[m])) if np.any(m) else 0.0
  area = float(np.sum(vol[m & (c1 > 0.1 * S0)]))
  print(f"[rg] DONE t={time:.0f}s in {niter} steps")
  print(f"[rg] plume reach = {reach:.0f} m,  footprint(>10% S0) = {area:.0f} m2,  "
        f"far-field dilution S0/c_min = {S0/max(cmin,1e-6):.1f}x")
  print(f"[rg] bottom salinity map -> VTK field c1 (ParaView)")
