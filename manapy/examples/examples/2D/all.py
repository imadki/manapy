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
from manapy.sys_solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.sys_solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.sys_solvers.advec.system import AdvectionSolver
from manapy.core.Variable import Variable
from manapy.backends.debug import log_step

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


dim, mesh_path, mesh_name = get_mesh("big/carre.msh")
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

ts = MPI.Wtime()

# TODO tfinal
if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .15
miter = 0
niter = 1
Pinit = 2.
saving_at_node = 1

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "neumann",
              "bottom": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
P = Variable(domain=domain, BC=boundaries, values_dict=values)


S = AdvectionDiffusionSolver(ne, vel=(u, v), Dxx=0.01, Dyy=0., order=2, cfl=0.8)
# S = AdvectionSolver(ne, vel=(u, v), order=1, cfl=0.8)

####Initialisation
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)

if RANK == 0: print("Start While loop ...")

# loop over time
while time < tfinal:

  # TODO -1
  u.face[:] = 2.
  v.face[:] = 0.

  log_step.log("u_interpolate_facetocell")
  u.interpolate_facetocell()
  log_step.out()

  log_step.log("v_interpolate_facetocell")
  v.interpolate_facetocell()
  log_step.out()

  log_step.log("S.stepper")
  d_t = S.stepper()
  log_step.out()
  tot = int(tfinal / d_t / 10) + 1

  time = time + d_t

  log_step.log("compute_fluxes")
  S.compute_fluxes()
  log_step.out()

  log_step.log("compute_new_val")
  S.compute_new_val()
  log_step.out()


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

      # domain.save_on_node_multi(d_t, time, niter, miter, variables=["ne", "u","v", "P"],
      #                          values=[ne.node, u.node,v.node, P.node], file_format="vtu")
    else:
      domain.save_on_cell_multi(d_t, time, niter, miter, variables=["ne", "u", "v", "P"],
                                values=[ne.cell, u.cell, v.cell, P.cell], file_format="vtu")
    miter += 1

  niter += 1

te = MPI.Wtime()


# End measuring energy after the function execution
tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)