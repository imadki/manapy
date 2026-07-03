#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOF phase-fraction advection test (interFoam phase 1): a circular blob of fluid 1
(alpha=1) in fluid 2 (alpha=0) is transported by a prescribed divergence-free
solid-body rotation. The bounded, conservative MULES (Zalesak-limited) scheme keeps
alpha in [0,1] and the interface sharp (interface compression, cAlpha) while
conserving the phase volume sum(alpha*V).

Compare cAlpha=0 (plain bounded upwind -> the interface diffuses) with cAlpha=1
(compression -> the interface stays sharp), both mass-conserving.

Run:
    python3 vof_advection2d.py                 # one quarter turn, cAlpha=1
    CALPHA=0 python3 vof_advection2d.py
    mpirun -np 4 python3 vof_advection2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.vof import VOFAdvection

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', 'meshes', 'geo')
mesh = os.environ.get('MESH', os.path.join(BASE, 'carre.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)

xc, yc = domain.cells.center[:, 0], domain.cells.center[:, 1]
# solid-body rotation about the box centre (divergence-free)
uc, vc = -(yc - 0.5), (xc - 0.5)
umax = 0.5

alpha = Variable(domain=domain)
alpha.cell[:] = np.where((xc - 0.5) ** 2 + (yc - 0.75) ** 2 < 0.15 ** 2, 1.0, 0.0)

cAlpha = float(os.environ.get('CALPHA', 1.0))
vof = VOFAdvection(alpha, cAlpha=cAlpha)
phi = vof.face_flux(uc, vc)                            # frozen div-free advecting flux

V0 = vof.phase_volume()
# global dt (min cell size reduced across ranks)
hmin2 = MPI.COMM_WORLD.allreduce(float(domain.cells.volume.min()), op=MPI.MIN)
dt = 0.2 * np.sqrt(hmin2) / umax
nstep = int(float(os.environ.get('TURNS', 0.25)) * 2 * np.pi / dt)

for it in range(nstep):
  vof.step(phi, dt)
  if it % 50 == 0:
    lo, hi = vof.bounds()
    if RANK == 0:
      print(f"it {it:5d}  alpha in [{lo:.4f}, {hi:.4f}]  mass={vof.phase_volume():.6f}")

lo, hi = vof.bounds(); V1 = vof.phase_volume()
nint = MPI.COMM_WORLD.allreduce(int(np.sum((alpha.cell > 0.01) & (alpha.cell < 0.99))), op=MPI.SUM)
domain.save_on_cell_multi(["alpha"], [alpha.cell], dt, 0.0, nstep, 0)
if RANK == 0:
  print(f"\nVOF advection ({nstep} steps, cAlpha={cAlpha})")
  print(f"  bounds        : [{lo:.6f}, {hi:.6f}]   (must stay in [0,1])")
  print(f"  phase volume  : {V0:.6f} -> {V1:.6f}   ({100 * (V1 - V0) / V0:+.3f}%)")
  print(f"  interface band: {nint} cells (0.01<alpha<0.99; smaller = sharper)")
  print("  wrote VTK (alpha) to ./vtk_results/")
