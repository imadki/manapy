#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-phase (VOF) lid-driven cavity: a dense drop (fluid 1) inside a light fluid (fluid 2)
is advected by the cavity flow. Exercises the variable-density/viscosity PISO coupled to
the bounded MULES phase-fraction transport (interFoam phase 2, no surface tension yet):

  rho(alpha) = alpha*rho1 + (1-alpha)*rho2,   mu(alpha) = alpha*mu1 + (1-alpha)*mu2

The momentum is assembled in conservative form (rho V/dt time term, mass-flux convection
rho_f phi_f, mu_f diffusion); alpha is transported each step by the divergence-free PISO
flux. alpha stays in [0,1] and the phase volume is conserved even at a 1000:1 density ratio.

Run:
    RHO1=1000 python3 vof_cavity2d.py
    mpirun -np 4 python3 vof_cavity2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.system import IncompressibleSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', 'meshes', 'geo')
mesh = os.environ.get('MESH', os.path.join(BASE, 'carre.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)

xc, yc = domain.cells.center[:, 0], domain.cells.center[:, 1]
U = 1.0
rho1 = float(os.environ.get('RHO1', 1000.0)); rho2 = 1.0     # drop / ambient
mu1, mu2 = 1e-2, 1e-4

u = Variable(domain=domain); v = Variable(domain=domain)
P = Variable(domain=domain,
             BC={"upper": "neumann", "in": "neumann", "out": "neumann", "bottom": "dirichlet"},
             values_dict={"bottom": 0.0})
alpha = Variable(domain=domain)
alpha.cell[:] = np.where((xc - 0.5) ** 2 + (yc - 0.6) ** 2 < 0.15 ** 2, 1.0, 0.0)

solver = IncompressibleSolver(u, v, P, ncorr=3, implicit_momentum=True,
                              u_bc={"upper": U, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              v_bc={"upper": 0.0, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              alpha=alpha, rho1=rho1, rho2=rho2, mu1=mu1, mu2=mu2, cAlpha=1.0)

hmin2 = MPI.COMM_WORLD.allreduce(float(domain.cells.volume.min()), op=MPI.MIN)
dt = float(os.environ.get('DT', 3.0 * 0.4 * np.sqrt(hmin2) / U))
V0 = solver.vof.phase_volume()
if RANK == 0:
  print(f"dt={dt:.4g}  rho ratio={rho1 / rho2:.0f}")

for it in range(int(os.environ.get('NSTEP', 400))):
  solver.step(dt=dt)
  if it % 100 == 0:
    lo, hi = solver.vof.bounds()
    umax = MPI.COMM_WORLD.allreduce(float(np.max(np.abs(u.cell))), op=MPI.MAX)
    if RANK == 0:
      print(f"it {it:5d}  |u|max={umax:.3f}  alpha in [{lo:.4f},{hi:.4f}]  mass={solver.vof.phase_volume():.6f}")

domain.save_on_cell_multi(["u", "v", "P", "alpha"], [u.cell, v.cell, P.cell, alpha.cell], dt, 0.0, 0, 0)
lo, hi = solver.vof.bounds(); V1 = solver.vof.phase_volume()
if RANK == 0:
  print(f"\nTwo-phase VOF cavity (rho ratio {rho1 / rho2:.0f})")
  print(f"  alpha bounds : [{lo:.6f}, {hi:.6f}]   (must stay in [0,1])")
  print(f"  phase volume : {V0:.6f} -> {V1:.6f}   ({100 * (V1 - V0) / V0:+.3f}%)")
  print("  wrote VTK (u, v, P, alpha) to ./vtk_results/")
