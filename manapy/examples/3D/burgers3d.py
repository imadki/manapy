#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit
from manapy.domain import Domain, Partitioning
from manapy.helpers import get_mesh
from manapy.solvers.advecdiff.tools_utils_compute import initialisation_gaussian_3d
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


dim, mesh_path, mesh_name = get_mesh("hybrid3d.msh")
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells


# TODO tfinal
if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .25
miter = 0
niter = 1
Pinit = 2.
saving_at_node = 1

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "neumann",
              "bottom": "neumann",
              "front": "neumann",
              "back": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
w = Variable(domain=domain)
P = Variable(domain=domain, BC=boundaries, values_dict=values)

# Call the transport solver
S = AdvectionDiffusionSolver(ne, vel=(u, v), Dxx=0.1, Dyy=0., Dzz=0., order=2, cfl=0.8)

####Initialisation
initialisation_gaussian_3d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

# loop over time
while time < tfinal:

  # TODO -1
  u.face[:] = 2.
  v.face[:] = 0.
  w.face[:] = 0.

  u.interpolate_facetocell()
  v.interpolate_facetocell()
  w.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      # save vtk files for the solution
      ne.update_halo_value()
      ne.update_ghost_value()
      ne.interpolate_celltonode()

      # save vtk files for the solution
      u.update_halo_value()
      u.update_ghost_value()
      u.interpolate_celltonode()

      # save vtk files for the solution
      v.update_halo_value()
      v.update_ghost_value()
      v.interpolate_celltonode()

      # save vtk files for the solution
      w.update_halo_value()
      w.update_ghost_value()
      w.interpolate_celltonode()

      domain.save_on_node_multi(d_t, time, niter, miter, variables=["ne", "u", "v", "w", "P"],
                                values=[ne.node, u.node, v.node, w.node, P.node])
    else:
      domain.save_on_cell_multi(d_t, time, niter, miter, variables=["ne", "u", "v", "w", "P"],
                                values=[ne.cell, u.cell, v.cell, w.cell, P.cell])
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

# del L
