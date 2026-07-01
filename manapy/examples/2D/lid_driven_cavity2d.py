#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lid-driven cavity with the incompressible projection solver (manapy's icoFoam-like).

Unit square, three no-slip walls and a lid (top) sliding at U=1; Reynolds number
Re = U L / nu. The face-flux-consistent Chorin projection is marched to steady state
and the vertical-centreline velocity profile u(y) at x=0.5 is compared to the
Ghia, Ghia & Shin (1982) benchmark. Density and (u,v) are written to VTK.

A coarse mesh reaches steady state in far fewer steps; the bundled carre.msh may be
fine (slow). Generate a coarse square with gmsh if needed, e.g.
    sed 's/lc = .01;/lc = .025;/' meshes/geo/carre.geo > /tmp/cav.geo
    gmsh /tmp/cav.geo -2 -format msh2 -o /tmp/cav.msh
    MESH=/tmp/cav.msh python3 lid_driven_cavity2d.py

Run:
    RE=100 python3 lid_driven_cavity2d.py
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

# The pressure Poisson solver (scheme='fv') is a free choice of backend -- swap in
# MUMPSSolver / GinkgoDistributedSolver here; PETSc CG is the default.
# from manapy.solvers.ls import MUMPSSolver
# poisson = MUMPSSolver(domain=domain, var=P, reuse_mtx=True, scheme='fv')
solver = IncompressibleSolver(u, v, P, nu=nu, rho=1.0, cfl=0.4,
                              u_bc={"upper": U, "bottom": 0.0, "in": 0.0, "out": 0.0},
                              v_bc={"upper": 0.0, "bottom": 0.0, "in": 0.0, "out": 0.0})

nmax = int(os.environ.get('NSTEP', 20000))
uold = u.cell.copy()
for it in range(nmax):
  solver.step()
  if it % 500 == 0:
    ch = float(np.max(np.abs(u.cell - uold))); uold = u.cell.copy()
    if RANK == 0 and it % 2000 == 0:
      print(f"it {it:6d}  |u|max={np.max(np.abs(u.cell)):.3f} |v|max={np.max(np.abs(v.cell)):.3f}"
            f"  div={solver.divergence_norm():.2e}  dU={ch:.2e}")
    if it > 0 and ch < 5e-5:
      if RANK == 0:
        print(f"steady at it={it}")
      break

domain.save_on_cell_multi(["u", "v", "P"], [u.cell, v.cell, P.cell], solver.dt, 0.0, it, 0)

# Ghia, Ghia & Shin (1982), u on the vertical centreline x=0.5 (Re=100)
gy = np.array([0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5, .6172, .7344,
               .8516, .9531, .9609, .9688, .9766, 1.])
gu = np.array([0, -.03717, -.04192, -.04775, -.06434, -.1015, -.15662, -.2109, -.20581,
               -.13641, .00332, .23151, .68717, .73722, .78871, .84123, 1.])
if RANK == 0:
  xc, yc = domain.cells.center[:, 0], domain.cells.center[:, 1]
  band = np.abs(xc - 0.5) < 0.02
  o = np.argsort(yc[band]); ys, us = yc[band][o], u.cell[band][o]
  ui = np.interp(gy, ys, us)
  print(f"\nLid-driven cavity Re={Re:.0f}  (u on x=0.5 vs Ghia et al. 1982)")
  print(f"  u_min = {us.min():.4f}  (Ghia -0.2109)")
  if Re == 100:
    print(f"  L2(profile vs Ghia) = {np.sqrt(np.mean((ui - gu) ** 2)):.4f}")
  print("  wrote VTK (u, v, P) to ./vtk_results/")
