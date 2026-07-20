#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D linear advection with PERIODIC boundaries.

A Gaussian blob is advected at a constant velocity (Uadv, Vadv) across a
doubly-periodic unit square: it leaves one side and re-enters the opposite side.
Runs in serial AND in parallel -- cross-rank periodic pairs are handled by the
domain's periodic halo (partner delivered as a translated halo cell), so no
solver change is needed.

Mesh requirement: a periodic-tagged square whose four sides carry the physical
tags in=11, out=22, upper=33, bottom=44 (that is what makes the domain build the
periodic connectivity). `meshes/geo/periodic_square.msh` is such a mesh; generate
one with manapy.api.meshgen + a `Periodic Line` gmsh directive if you need
another resolution.

Run:
    python advection2d_periodic.py
    mpirun -n 4 python advection2d_periodic.py
Env overrides: MESH, TFINAL, U, V.
"""
from mpi4py import MPI
import os
from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.solvers.advec.system import AdvectionSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# -------------------------------------------------------------------- domain
try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  MESH_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'meshes', 'geo')
filename = os.environ.get('MESH', 'periodic_square.msh')  # sides tagged 11/22/33/44
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells

# All four sides periodic. NB: like the original advection2d.py, the advected
# variable needs no explicit BC -- the periodicity lives in the DOMAIN (built
# from the 11/22/33/44 mesh tags); this dict just documents the intent.
boundaries = {"in": "periodic", "out": "periodic",
              "upper": "periodic", "bottom": "periodic"}

ne = Variable(domain=domain)   # advected scalar
u  = Variable(domain=domain)
v  = Variable(domain=domain)
P  = Variable(domain=domain)

S = AdvectionSolver(ne, vel=(u, v), order=1, cfl=0.8)

# Gaussian blob + constant velocity. On the unit square a diagonal velocity
# (1,1) brings the blob back to its start after t=1 (one wrap in each direction).
Pinit = 2.0
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)

Uadv = float(os.environ.get('U', 1.0))
Vadv = float(os.environ.get('V', 0.0))
tfinal = float(os.environ.get('TFINAL', 1.0))

time = 0.0
niter = 1
miter = 0
d_t = 0.0
if RANK == 0:
  print(f"periodic advection: vel=({Uadv},{Vadv})  tfinal={tfinal}  ranks={COMM.Get_size()}")

# ----------------------------------------------------------------- time loop
while time < tfinal:
  u.face[:] = Uadv
  v.face[:] = Vadv
  u.interpolate_facetocell()
  v.interpolate_facetocell()

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
