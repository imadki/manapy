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
from manapy.solvers.ls import PETScKrylovSolver, MUMPSSolver, ScipySolver
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

filename = 'hybrid3d.msh'
dim = 3
mesh_path = os.path.join(MESH_DIR, filename)
domain = Domain.create_domain(mesh_path, dim, recreate=True)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

if RANK == 0: print("Start Computation")
miter = 0
niter = 1
Pinit = 10.
saving_at_node = 1

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "dirichlet",
              "bottom": "dirichlet",
              "front": "dirichlet",
              "back": "dirichlet"
              }
values = {"in": 20,
          "out": 0.,
          "upper": 0.,
          "bottom": 0.,
          "front": 0.,
          "back": 0.
          }

P = Variable(domain=domain, BC=boundaries, values_dict=values)


L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond',
              precond='gamg', sub_precond="amg",
              eps_a=1e-10, eps_r=1e-10, method="gmres")

ts = MPI.Wtime()
L()
P.update_halo_value()
P.update_ghost_value()
P.interpolate_celltonode()

domain.save_on_node_multi(variables=["P"], values=[P.node], dt=0., time=0., niter=niter, miter=miter)

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

