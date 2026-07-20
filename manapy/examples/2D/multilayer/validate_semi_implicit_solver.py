#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate SemiImplicitMLSolver (the solver-class semi-implicit) on the manapy mesh.

(1) N=1 barotropic standing wave vs EXACT, at 10x the explicit CFL.
(2) N=2 internal seiche (dense bottom + ambient): the interface returns after one
    baroclinic period, run at several x the barotropic CFL (explicit would blow up).
Both on a TRIANGLE mesh -> uses the Diamond Laplacian.
"""
from mpi4py import MPI
import numpy as np
from manapy.api.mesh import Mesh
from manapy.solvers.multilayer.semi_implicit import SemiImplicitMLSolver

RANK = MPI.COMM_WORLD.Get_rank()
G = 9.81
mesh = Mesh.rectangle(bounds=((0., 1.), (0., 1.)), n=(48, 48), cell_type="triangle")
dom = mesh.domain
cc = np.asarray(dom.cells.center)
cv = np.asarray(dom.cells.volume)
xc = cc[:, 0]
ncell = dom.nbcells
dx = 1.0 / 48
KX = np.pi
A0 = 1e-3


def l2(a, b):
  return np.sqrt(np.sum((a - b)**2 * cv) / (np.sum(b**2 * cv) + 1e-30))


# ---------- (1) N=1 barotropic standing wave -------------------------------
H = 1.0
c0 = np.sqrt(G * H)
omega = c0 * KX
period = 2 * np.pi / omega
dt_cfl = 0.9 * dx / c0
dt = 10 * dt_cfl
S1 = SemiImplicitMLSolver(dom, rho=[1000.], Href=[H], dt=dt, grav=G, scheme="diamond")
st = {"eta": A0 * np.cos(KX * xc), "u": [np.zeros(ncell)], "v": [np.zeros(ncell)]}
t = 0.0; emax = 0.0
T = 2 * period
while t < T - 1e-12:
  S1.step(st); t += dt; emax = max(emax, np.max(np.abs(st["eta"])))
exact = A0 * np.cos(KX * xc) * np.cos(omega * T)
e1 = l2(st["eta"], exact)
if RANK == 0:
  print(f"[SI-solver] (1) N=1 standing wave, dt={dt/dt_cfl:.0f}x CFL: stable(max/A0={emax/A0:.2f}) "
        f"L2 vs EXACT = {e1:.3e} -> {'PASS' if e1 < 0.05 else 'FAIL'}")


# ---------- (2) N=2 internal seiche ----------------------------------------
RHO1, RHO2 = 1030., 1000.
H1, H2 = 0.5, 0.5
gp = G * (RHO1 - RHO2) / RHO2
c1 = np.sqrt(gp * H1 * H2 / (H1 + H2))
T1 = 2 * np.pi / (c1 * KX)
dt_baro = 0.9 * dx / np.sqrt(G * (H1 + H2))
dt2 = 2 * dt_baro                                  # stable regime (see note below)
S2 = SemiImplicitMLSolver(dom, rho=[RHO1, RHO2], Href=[H1, H2], dt=dt2, grav=G, scheme="diamond")
ei0 = A0 * np.cos(KX * xc)
st2 = {"eta": np.zeros(ncell), "eta_i": ei0.copy(),
       "u": [np.zeros(ncell), np.zeros(ncell)], "v": [np.zeros(ncell), np.zeros(ncell)]}
t = 0.0; emax = 0.0
while t < T1 - 1e-12:
  S2.step(st2); t += dt2; emax = max(emax, np.max(np.abs(st2["eta_i"])))
e2 = l2(st2["eta_i"], ei0)
if RANK == 0:
  print(f"[SI-solver] (2) N=2 seiche c0/c1={np.sqrt(G*(H1+H2))/c1:.1f}, dt={dt2/dt_baro:.0f}x baro CFL: "
        f"stable(max/A0={emax/A0:.2f}) interface return L2={e2:.3e} -> {'PASS' if e2 < 0.1 else 'FAIL'}")
  print(f"[SI-solver] (note) N=2 stable to ~2.5x barotropic CFL; a residual collocated")
  print(f"[SI-solver]        barotropic-baroclinic coupling caps the step below the full c0/c1.")
  print(f"[SI-solver] => semi-implicit multilayer integrated in the solver class, on the manapy Diamond mesh")
