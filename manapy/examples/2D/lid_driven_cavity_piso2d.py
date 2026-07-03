#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lid-driven cavity with the TRUE (implicit-momentum) PISO -- manapy's faithful
icoFoam. Same setup as lid_driven_cavity2d.py (unit square, lid at U=1, Re=U L/nu),
but the momentum equation is assembled and solved implicitly (a_P/H split +
Rhie-Chow face flux), so the time step is limited by accuracy, not by an explicit
CFL. Here dt is set ~10x the explicit stability limit: the explicit projection
(implicit_momentum=False) blows up at this dt; the implicit PISO stays stable and
converges to the same Ghia, Ghia & Shin (1982) profile.

Run:
    RE=100 python3 lid_driven_cavity_piso2d.py
    MESH=/tmp/cav.msh DT=0.1 python3 lid_driven_cavity_piso2d.py
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

Re = float(os.environ.get('RE', 100.0)); U = 1.0; L = 1.0
nu = U * L / Re

u = Variable(domain=domain); v = Variable(domain=domain)
# pressure: Neumann walls + one Dirichlet reference (bottom) to fix the constant.
P = Variable(domain=domain,
             BC={"upper": "neumann", "in": "neumann", "out": "neumann", "bottom": "dirichlet"},
             values_dict={"bottom": 0.0})

# implicit_momentum=True -> true PISO. The momentum linear solver (non-symmetric,
# implicit convection) and the pressure Poisson (scheme='fv', variable coefficient
# rebuilt each step) are both a free choice of manapy backend: backend='mumps'
# (direct, default, ~2.8x faster here) or 'petsc' (Krylov). Pass poisson=/momentum=
# to bring your own. Set with BACKEND=petsc python3 ...
solver = IncompressibleSolver(u, v, P, nu=nu, rho=1.0, ncorr=3,
                              implicit_momentum=True, mom_predictor=True,
                              backend=os.environ.get('BACKEND', 'mumps'),
                              conv_order=int(os.environ.get('CONV', 1)),
                              u_bc={"upper": U, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              v_bc={"upper": 0.0, "bottom": 0.0, "in": 0.0, "out": 0.0})

# a large fixed dt (the whole point of implicit momentum); ~10x the explicit CFL.
# dt MUST be global (same on every rank): cells.volume.min() is per-rank in MPI, so
# reduce it -- otherwise each rank would assemble V/dt with a different dt.
hmin2 = MPI.COMM_WORLD.allreduce(float(domain.cells.volume.min()), op=MPI.MIN)
h = float(np.sqrt(hmin2))
dt = float(os.environ.get('DT', 10.0 * 0.4 * h / U))
if RANK == 0:
  print(f"dt = {dt:.4g}  (explicit CFL limit ~ {0.4 * h / U:.4g})")

nmax = int(os.environ.get('NSTEP', 20000))
uold = u.cell.copy()
for it in range(nmax):
  solver.step(dt=dt)
  if it % 100 == 0:
    # global steady criterion: reduce the per-rank change so every rank breaks
    # together (a per-rank local `ch` would let ranks diverge / deadlock in MPI).
    ch = MPI.COMM_WORLD.allreduce(float(np.max(np.abs(u.cell - uold))), op=MPI.MAX)
    uold = u.cell.copy()
    umax = MPI.COMM_WORLD.allreduce(float(np.max(np.abs(u.cell))), op=MPI.MAX)
    d = solver._mom_divergence(solver._phi)
    fdiv = np.sqrt(MPI.COMM_WORLD.allreduce(float(np.sum(d * d * solver.vol)), op=MPI.SUM))
    if RANK == 0:
      print(f"it {it:6d}  |u|max={umax:.3f}  fluxdiv={fdiv:.2e}  dU={ch:.2e}")
    if it > 0 and ch < 5e-5:
      if RANK == 0:
        print(f"steady at it={it}")
      break

domain.save_on_cell_multi(["u", "v", "P"], [u.cell, v.cell, P.cell], solver.dt, 0.0, it, 0)

# Ghia, Ghia & Shin (1982), u on the vertical centreline x=0.5 (Re=100). In MPI each
# rank holds only its partition, so gather the centreline band to rank 0 first.
gy = np.array([0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5, .6172, .7344,
               .8516, .9531, .9609, .9688, .9766, 1.])
gu = np.array([0, -.03717, -.04192, -.04775, -.06434, -.1015, -.15662, -.2109, -.20581,
               -.13641, .00332, .23151, .68717, .73722, .78871, .84123, 1.])
xc, yc = domain.cells.center[:, 0], domain.cells.center[:, 1]
band = np.abs(xc - 0.5) < 0.02
ys = np.concatenate(MPI.COMM_WORLD.allgather(yc[band]))
us = np.concatenate(MPI.COMM_WORLD.allgather(u.cell[band]))
if RANK == 0:
  o = np.argsort(ys); ys, us = ys[o], us[o]
  ui = np.interp(gy, ys, us)
  print(f"\nLid-driven cavity Re={Re:.0f} (TRUE PISO, u on x=0.5 vs Ghia et al. 1982)")
  print(f"  u_min = {us.min():.4f}  (Ghia -0.2109)")
  if Re == 100:
    print(f"  L2(profile vs Ghia) = {np.sqrt(np.mean((ui - gu) ** 2)):.4f}")
  print("  wrote VTK (u, v, P) to ./vtk_results/")
