#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from manapy.partitions import MeshPartition
from manapy.ddm import Domain

from manapy.solvers.advecdiff.tools_utils import initialisation_gaussian_3d
from manapy.solvers.advecdiff import AdvectionDiffusionSolver

from manapy.ast import Variable
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver
from manapy.base.base import Struct


import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "cube_bis.msh"

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 3
running_conf = Struct(backend="numba", signature=True, cache=True)
mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim, conf=running_conf)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

end = timeit.default_timer()

tt = COMM.reduce(end -start, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to create the domain", tt)

#TODO tfinal
if RANK == 0: print("Start Computation")
miter = 0
niter = 1
Pinit = 10.
saving_at_node = 1

boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann",
              "front":"neumann",
              "back":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }


P  = Variable(domain=domain, BC=boundaries, values=values)

conf = Struct(reuse_mtx=True, scheme='diamond', verbose=False)
L = PETScKrylovSolver(domain=domain, var=P, conf=conf)

ts = MPI.Wtime()

L()
P.update_halo_value()
P.update_ghost_value()
P.interpolate_celltonode()

domain.save_on_node_multi(0., 0., niter, miter, variables=["P"],values=[P.node])
       

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)

