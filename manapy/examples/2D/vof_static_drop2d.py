#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static-drop / Laplace-law test for the VOF surface tension (interFoam phase 3).

A circular drop of fluid 1 (radius R) sits in fluid 2 with equal density and no gravity;
only surface tension sigma acts. At equilibrium the CSF (Continuum Surface Force) model
must produce the Laplace pressure jump across the interface (2D):

    Delta p = sigma / R

The surface tension is well-balanced (a face flux D_f sigma K_f snGrad(alpha) added to
phiHbyA + the cell force sigma K grad(alpha)), the curvature K = -div(n_hat) with the
interface normal smoothed a few passes (much less noisy discrete curvature). This test
checks the pressure jump and reports the residual spurious ("parasitic") currents.

Run:
    R=0.2 SIGMA=1 python3 vof_static_drop2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.system import IncompressibleSolver

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.environ.get('MESH', os.path.join(BASE, 'carre.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)

xc, yc = domain.cells.center[:, 0], domain.cells.center[:, 1]
R = float(os.environ.get('R', 0.2)); sigma = float(os.environ.get('SIGMA', 1.0))

u = Variable(domain=domain); v = Variable(domain=domain)
P = Variable(domain=domain,
             BC={"upper": "neumann", "in": "neumann", "out": "neumann", "bottom": "dirichlet"},
             values_dict={"bottom": 0.0})
alpha = Variable(domain=domain)
r = np.sqrt((xc - 0.5) ** 2 + (yc - 0.5) ** 2)
alpha.cell[:] = 0.5 * (1.0 - np.tanh((r - R) / 0.02))  # smooth circle (~1 cell interface)

# equal density isolates surface tension; no compression (cAlpha=0) so the drop keeps
# its shape; walls all round.
solver = IncompressibleSolver(u, v, P, ncorr=3, implicit_momentum=True,
                              u_bc={"upper": 0.0, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              v_bc={"upper": 0.0, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              alpha=alpha, rho1=1.0, rho2=1.0, mu1=0.05, mu2=0.05,
                              cAlpha=0.0, sigma=sigma)
solver.nsmooth = int(os.environ.get('NSMOOTH', 4))     # curvature-normal smoothing passes

h = np.sqrt(MPI.COMM_WORLD.allreduce(float(domain.cells.volume.min()), op=MPI.MIN))
dt = 0.3 * np.sqrt(1.0 * h ** 3 / (2 * np.pi * sigma))  # capillary time-step limit

for it in range(int(os.environ.get('NSTEP', 400))):
  solver.step(dt=dt)

# pressure jump and spurious current (reduced across ranks)
pin_l = alpha.cell > 0.9; pout_l = alpha.cell < 0.1
comm = MPI.COMM_WORLD
pin = comm.allreduce(float(P.cell[pin_l].sum()), MPI.SUM) / max(comm.allreduce(int(pin_l.sum()), MPI.SUM), 1)
pout = comm.allreduce(float(P.cell[pout_l].sum()), MPI.SUM) / max(comm.allreduce(int(pout_l.sum()), MPI.SUM), 1)
umax = comm.allreduce(float(np.max(np.abs(u.cell))), MPI.MAX)
domain.save_on_cell_multi(["u", "v", "P", "alpha"], [u.cell, v.cell, P.cell, alpha.cell], dt, 0.0, 0, 0)
if RANK == 0:
  print(f"\nStatic drop (Laplace) R={R}, sigma={sigma}")
  print(f"  Delta p   = {pin - pout:.4f}   (exact sigma/R = {sigma / R:.4f})")
  print(f"  spurious |u|max = {umax:.2e}   (parasitic currents; -> 0 for exact balance)")
  print("  wrote VTK (u, v, P, alpha) to ./vtk_results/")
