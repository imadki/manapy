#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit
import os
from manapy.domain import Domain, Partitioning
from manapy.solvers.advecdiff.tools_utils_compute import initialisation_gaussian_3d
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.solvers.ls import PETScKrylovSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes')

filename = 'hybrid3d.msh'
dim = 3
mesh_path = os.path.join(MESH_DIR, filename)
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
S = AdvectionDiffusionSolver(ne, vel=(u, v), Dxx=0., Dyy=0., order=2, cfl=0.8)

####Initialisation
initialisation_gaussian_3d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)

###Linear sys confi###
# If you want the default options please do conf = Struct()
# reuse_mtx: matrix does not change during the while loop
# scheme: diamond or fv
# verbose: printing the mumps/petsc output
L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond',
              precond='gamg', sub_precond="amg",
              eps_a=1e-10, eps_r=1e-10, method="gmres")

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

# loop over time
while time < tfinal:

  L()
  P.update_halo_value()
  P.update_ghost_value()
  P.interpolate_celltonode()
  L.compute_Sol_gradient()

  # TODO -1
  u.face[:] = P.gradfacex[:]
  v.face[:] = P.gradfacey[:]
  w.face[:] = P.gradfacez[:]

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

      domain.save_on_node_multi(["ne", "u", "v", "w", "P"],
                                [ne.node, u.node, v.node, w.node, P.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "u", "v", "w", "P"],
                                [ne.cell, u.cell, v.cell, w.cell, P.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

# del L
