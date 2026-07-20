#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D linear advection with PERIODIC boundaries (the 3D twin of
examples/2D/advection/advection2d_periodic.py).

A Gaussian blob is advected at a constant velocity (U, V, W) across a
triply-periodic unit cube: it leaves one face and re-enters the opposite one, in
every direction. Runs in serial AND in parallel -- cross-rank periodic pairs are
delivered as translated halo cells, same as in 2D.

Mesh requirement: a periodic-tagged cube whose six faces carry the physical tags
in=11(x=0) out=22(x=Lx) upper=33(y=Ly) bottom=44(y=0) front=55(z=0) back=66(z=Lz),
AND is conforming across opposite faces. A structured HEX cube works
(`meshes/periodic_cube_hex.msh`, gmsh transfinite + Recombine + Periodic Surface);
tetrahedral transfinite cubes are currently rejected by the mesh reader.

Run:
    python advection3d_periodic.py
    mpirun -n 4 python advection3d_periodic.py
Env overrides: MESH, TFINAL, U, V, W.
"""
from mpi4py import MPI
import os
from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_3d
from manapy.solvers.advec.system import AdvectionSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# -------------------------------------------------------------------- domain
try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  MESH_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'meshes')
filename = os.environ.get('MESH', 'periodic_cube_hex.msh')  # 6 faces tagged 11..66
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, 3, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells

# All six faces periodic. Like the 2D example, the advected variable carries no
# explicit BC -- the periodicity lives in the DOMAIN (built from the 11..66 tags).
boundaries = {"in": "periodic", "out": "periodic",
              "upper": "periodic", "bottom": "periodic",
              "front": "periodic", "back": "periodic"}

ne = Variable(domain=domain)   # advected scalar
u  = Variable(domain=domain)
v  = Variable(domain=domain)
w  = Variable(domain=domain)
P  = Variable(domain=domain)

S = AdvectionSolver(ne, vel=(u, v, w), order=1, cfl=0.8)

# Gaussian blob + constant velocity. On the unit cube a diagonal velocity
# (1,1,1) brings the blob back to its start after t=1 (one wrap per direction).
Pinit = 2.0
initialisation_gaussian_3d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit)

Uadv = float(os.environ.get('U', 1.0))
Vadv = float(os.environ.get('V', 1.0))
Wadv = float(os.environ.get('W', 1.0))
tfinal = float(os.environ.get('TFINAL', 1.0))

time = 0.0
niter = 1
miter = 0
d_t = 0.0
if RANK == 0:
  print(f"periodic 3D advection: vel=({Uadv},{Vadv},{Wadv})  tfinal={tfinal}  ranks={COMM.Get_size()}")

# ----------------------------------------------------------------- time loop
while time < tfinal:
  u.face[:] = Uadv
  v.face[:] = Vadv
  w.face[:] = Wadv
  u.interpolate_facetocell()
  v.interpolate_facetocell()
  w.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1
  time += d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    ne.update_halo_value()
    ne.update_ghost_value()
    ne.interpolate_celltonode()
    domain.save_on_node_multi(["ne"], [ne.node], d_t, time, niter, miter)
    miter += 1
  niter += 1

if RANK == 0:
  print(f"done: {niter} steps, tfinal={time:.3f}")
