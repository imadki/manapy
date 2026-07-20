#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit, os

from manapy.domain import Domain, Partitioning
import numpy as np
from manapy.core.Variable import Variable
from manapy.solvers.ls import (MUMPSSolver, PETScKrylovSolver, ScipySolver, 
                               GinkgoSolver, GinkgoDistributedSolver)


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

start = timeit.default_timer()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = "carre.msh"
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

miter = 0
niter = 1
Pinit = 10.
saving_at_node = 1

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "dirichlet",
              "bottom": "dirichlet",
              }
values = {"in": 20,
          "out": 0.,
          "upper": 0.,
          "bottom": 0.,
          }

P = Variable(domain=domain, BC=boundaries, values_dict=values)


L = GinkgoDistributedSolver(domain=domain, var=P, reuse_mtx=True,
                            precond="jacobi", scheme='diamond',
                            device="cpu"
                            )

ts = MPI.Wtime()
L()
te = MPI.Wtime()

P.update_halo_value()
P.update_ghost_value()
P.interpolate_celltonode()

domain.save_on_node_multi(variables=["P"], values=[P.node], dt=0., time=0., niter=niter, miter=miter)


tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)
