#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realistic SWRO desalination brine outfall -- far-field footprint (industrial demo).

An outfall on a sloping coastal shelf continuously delivers a dense brine layer
(h0, S0) -- the OUTPUT of a near-field model such as CORMIX (see brine_plume_hllc2d
for the physics). The far-field solver (HLLC, transcritical-robust, + turbulent
entrainment) then predicts the dense plume plunging offshore down the slope, mixing
with ambient seawater, and the resulting BOTTOM HYPERSALINITY FOOTPRINT.

Dimensional, plant-scale parameters. Writes the bottom excess-salinity map c1 to VTK.

  near-field (CORMIX)                 THIS far-field solver
  --> delivers (h0, S0) at the -->    plunge + entrainment + spreading
      outfall on the seabed           --> bottom salinity footprint & dilution
"""
from mpi4py import MPI
import numpy as np
from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.system import MultilayerSWSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

# --- plant-scale parameters -------------------------------------------------
LX, LY = 300.0, 80.0            # coastal domain (m)
SLOPE = 0.03                    # 3% shelf slope (deepens offshore, +x)
D0 = 10.0                       # water depth at the shoreline edge (m)
RHO_AMB = 1027.0               # ambient seawater density
RHO_BRINE = 1035.0            # dense layer density at the bed (after near-field dilution)
GRAV = 9.81
gp = GRAV * (RHO_BRINE - RHO_AMB) / RHO_AMB
H_FILM = 0.05                  # thin ambient residual dense film (m)
X_OUT, Y_OUT, R_OUT = 30.0, 40.0, 8.0   # outfall location + footprint radius (m)
H0 = 0.5                       # dense-layer thickness delivered by the diffuser (m)
S0 = 1.0                       # source excess-salinity signal (normalised; 1 = full brine excess)
MANN = 0.025                   # bottom drag
TFIN = 400.0                   # seconds

mesh = Mesh.rectangle(bounds=((0., LX), (0., LY)), n=(150, 40), cell_type="triangle")
domain = mesh.domain
cc = np.asarray(domain.cells.center)
xc, yc = cc[:, 0], cc[:, 1]
vol = np.asarray(domain.cells.volume)

Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)         # seabed
depth = D0 + SLOPE * xc                                       # total water depth
outfall = ((xc - X_OUT)**2 + (yc - Y_OUT)**2) < R_OUT**2      # (kept for diagnostics)
w_src = np.exp(-((xc - X_OUT)**2 + (yc - Y_OUT)**2) / R_OUT**2)   # SMOOTH diffuser footprint

# start as a thin uniform film; the (smooth) source builds the plume up -> no initial
# discontinuity, so no grid-scale oscillations seeded at the outfall.
h1_0 = np.full(len(xc), H_FILM)
layers = [
  {'h':  mesh.field("h1", init=h1_0, bc=NEU), 'hu': mesh.field("hu1", init=0., bc=NEU),
   'hv': mesh.field("hv1", init=0., bc=NEU), 's': mesh.field("s1", init=0., bc=NEU)},
  {'h':  mesh.field("h2", init=depth - h1_0, bc=NEU), 'hu': mesh.field("hu2", init=0., bc=NEU),
   'hv': mesh.field("hv2", init=0., bc=NEU), 's': mesh.field("s2", init=0., bc=NEU)},
]
# SRNH here (not HLLC): this mild-slope radial spread is subcritical, so SRNH's Roe
# diffusion damps the gravity-current front and avoids the HLLC bed-coupling oscillations
# (keep HLLC only for a steep, strongly-plunging plume).
S = MultilayerSWSolver(layers, rho=[RHO_BRINE, RHO_AMB], Z=Z, grav=GRAV, cfl=0.8, order=1,
                       scheme="srnh", entrain=True, E0=0.075, Mann=MANN,
                       Ci=0.02, ri_crit=1.0)          # light interfacial friction (safety)

if RANK == 0:
  print(f"[outfall] SWRO shelf {LX}x{LY} m, slope {SLOPE*100:.0f}%, g'={gp:.3f}, "
        f"brine {RHO_BRINE}/amb {RHO_AMB} kg/m3")
  print(f"[outfall] diffuser holds h0={H0} m, S0={S0} at ({X_OUT},{Y_OUT}) m; running {TFIN}s ...")

h1c = np.asarray(layers[0]['h'].cell); s1c = np.asarray(layers[0]['s'].cell)
time, niter, miter = 0.0, 0, 0
while time < TFIN:
  dt = S.stepper(); time += dt
  S.compute_fluxes(); S.compute_new_val()
  # steady source (SMOOTH): gently relax h1,s1 toward (h0, S0*h0) over the Gaussian
  # footprint. A hard clamp on a sharp disk seeds the grid-scale oscillations seen in h1.
  h1c[:] += w_src * (H0 - h1c) * 0.15
  s1c[:] += w_src * (S0 * H0 - s1c) * 0.15
  niter += 1
  if not np.all(np.isfinite(h1c)):
    if RANK == 0: print(f"[outfall] non-finite at t={time:.1f}");
    break
  if niter % 1000 == 0:
    c1 = s1c / np.maximum(h1c, 1e-9)
    domain.save_on_cell_multi(["h1", "c1", "Z"], [layers[0]['h'].cell, c1, Z.cell], dt, time, niter, miter)
    miter += 1
    if RANK == 0:
      m = h1c > 2 * H_FILM
      xr = np.max(xc[m]) if np.any(m) else X_OUT
      print(f"  t={time:5.1f}s  plume reach x={xr:5.1f}m  min conc in plume={np.min((s1c/np.maximum(h1c,1e-9))[m]):.3f}")

if RANK == 0:
  c1 = s1c / np.maximum(h1c, 1e-9)
  m = h1c > 2 * H_FILM
  # footprint: area where the dense layer carries >10% of source excess salinity
  hyper = m & (c1 > 0.1 * S0)
  area = float(np.sum(vol[hyper]))
  reach = float(np.max(xc[m]) - X_OUT) if np.any(m) else 0.0
  cmin = float(np.min(c1[m])) if np.any(m) else 0.0
  dilution = S0 / max(cmin, 1e-6)
  print(f"[outfall] DONE t={time:.0f}s")
  print(f"[outfall] plume reach down-slope = {reach:.0f} m")
  print(f"[outfall] hypersalinity footprint (>10% S0) area = {area:.0f} m^2")
  print(f"[outfall] far-field extra dilution S0/c_min = {dilution:.1f}x  (c_min={cmin:.3f})")
  print(f"[outfall] -> bottom salinity map written to VTK (field c1) for ParaView")
