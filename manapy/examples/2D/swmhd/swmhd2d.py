#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Shallow-Water MHD (SWMHD) example: rotating MHD vortex (choix = 40).

Run in serial:
    python swmhd2d.py
or in parallel:
    mpirun -n 4 python swmhd2d.py
"""
from mpi4py import MPI
from manapy.solvers.swmhd.system import ShallowWaterMHDSolver
from manapy.solvers.swmhd.tools_utils_compute import initialisation_SWMHD
import timeit
import os
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes')

filename = 'big/carre.msh'
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

end = timeit.default_timer()

tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

if RANK == 0:
  print("Start Computation ...")

time = 0
# NOTE: choix=41 is a rotating MHD vortex re-centred on (0.5,0.5) to fit the
# unit-square mesh. It is a steady state on an infinite/periodic domain; with
# the reflecting (neumann) walls used here it stays clean over this window but
# eventually destabilises near the walls. For long-time runs use a periodic
# setup or a larger domain. (Same behaviour as the legacy SWMHD model.)
tfinal = .03
miter = 0
niter = 1
saving_at_node = 1

boundaries = {"in": "neumann",
              "out": "neumann",
              "upper": "neumann",
              "bottom": "neumann"
              }

h = Variable(domain=domain)
hu = Variable(domain=domain, BC=boundaries)
hv = Variable(domain=domain, BC=boundaries)
hB1 = Variable(domain=domain, BC=boundaries)
hB2 = Variable(domain=domain, BC=boundaries)
PSI = Variable(domain=domain)
Z = Variable(domain=domain)

# choix = 41 : rotating MHD vortex centred on (0.5, 0.5). k1, k2, eps, tol unused.
initialisation_SWMHD(h.cell, hu.cell, hv.cell, hB1.cell, hB2.cell, PSI.cell, Z.cell,
                     cells.center, 7, 1.0, 1.0, 0.0, 0.0)

# GLM=10 enables the PSI relaxation (divergence cleaning) in update_SWMHD.
S = ShallowWaterMHDSolver(h=h, hvel=(hu, hv), hB=(hB1, hB2), PSI=PSI, Z=Z,
                          order=1, cfl=0.8, grav=1.0, GLM=10)

ts = MPI.Wtime()
if RANK == 0:
  print("Start While loop ...")

# loop over time
while time < tfinal:

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      for var in (h, hu, hv, hB1, hB2):
        var.update_halo_value()
        var.update_ghost_value()
        var.interpolate_celltonode()

      domain.save_on_node_multi(["h", "hu", "hv", "hB1", "hB2"],
                                [h.node, hu.node, hv.node, hB1.node, hB2.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["h", "hu", "hv", "hB1", "hB2"],
                                [h.cell, hu.cell, hv.cell, hB1.cell, hB2.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)
  print("Number of iterations", niter)
