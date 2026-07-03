#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit
import os
import numpy as np
from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver

from manapy.solvers.ls import (MUMPSSolver, PETScKrylovSolver, GinkgoDistributedSolver,
                               GinkgoSolver)

from manapy.core.Variable import Variable
from manapy.backends.gpu import GPUBackend
from manapy.backends.gpu import GPUArray


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes','geo')

filename = 'carre.msh'
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=True)
gpu.init_stream()
gpu.set_config(free=True)

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True,
                                backend=gpu
                               )
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
              "bottom": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
P = Variable(domain=domain, BC=boundaries, values_dict=values)

# Call the transport solver
S = AdvectionDiffusionSolver(ne, vel=(u, v), Dxx=0., Dyy=0., order=2, cfl=0.8)

####Initialisation
# Init sur HOST (lambda/np) puis copie explicite vers les champs (le backend du
# domaine decide : device sous GPU, host sous CPU).
be = domain.backend
_ne = np.zeros(nbcells); _u = np.zeros(nbcells); _v = np.zeros(nbcells); _P = np.zeros(nbcells)
initialisation_gaussian_2d(_ne, _u, _v, _P, be.to_host(cells.center), Pinit)
be.copy(ne.cell, _ne); be.copy(u.cell, _u)
be.copy(v.cell, _v); be.copy(P.cell, _P)
f = lambda x, y, z: Pinit * (1. - x)


###Linear sys confi###
# If you want the default options please do conf = Struct()
# reuse_mtx: matrix does not change during the while loop
# scheme: diamond or fv
# verbose: printing the mumps/petsc output
L = GinkgoDistributedSolver(domain=domain, var=P,
                            # precond="jacobi",
                            device="cuda",
                            scheme='diamond',
                            method="cg",
                            verbose=False,
                            reuse_mtx=True
                            )

# L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond',
#               precond='gamg', sub_precond="amg",  # with_mtx=False,
#               eps_a=1e-10, eps_r=1e-10, method="gmres")

# L = MUMPSSolver(domain=domain, var=P, 
#                 reuse_mtx=True,
#                 # precond="jacobi", 
#                 scheme='diamond')


ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

# loop over time
while time < tfinal:

  # print(f"[darcy rank {RANK}] before L iter={niter}", flush=True)
  L()
  # print("Max P", P.cell.max())
  # print(f"[darcy rank {RANK}] after L iter={niter}", flush=True)
  P.update_halo_value()
  P.update_ghost_value()
  P.interpolate_celltonode()
  L.compute_Sol_gradient()
  
  # print("Max P grad", P.gradfacex.max())

  # u.face <- grad(P) : copie explicite (device->device sous GPU, host sous CPU).
  be.copy(u.face, P.gradfacex)
  be.copy(v.face, P.gradfacey)

  u.interpolate_facetocell()
  v.interpolate_facetocell()

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

      # .node a ete calcule sur device par interpolate_celltonode : transfert
      # explicite vers host pour la sauvegarde (qui lit des arrays host).
      domain.save_on_node_multi(
        ["ne", "u", "v", "P"],
        [be.to_host(ne.node), be.to_host(u.node),
         be.to_host(v.node), be.to_host(P.node)], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "u", "v", "P"],
                                [ne.cell, u.cell, v.cell, P.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

# del L
