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
import numpy as np
from manapy.core.Variable import Variable
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver, ScipySolver


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

start = timeit.default_timer()

filename = "big/carre.msh"
dim, mesh_path, mesh_name = get_mesh(filename)
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

# TODO tfinal
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


# L = MUMPSSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond')

L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond',
              precond='gamg', sub_precond="amg",  # with_mtx=False,
              eps_a=1e-10, eps_r=1e-10, method="gmres")

# L = ScipySolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond')

# Solve the system
ts = MPI.Wtime()
L()
te = MPI.Wtime()

L.view()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)
