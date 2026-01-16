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
import numpy as np
from manapy.ast import Variable
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver
from manapy.base.base import Struct
from manapy.ast.functions2d import Mat_Assembly, Vec_Assembly

import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..','..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
dim = 2
filename = "rectangle.msh"
# filename = "mid_rectangle.msh"
# filename = "bigger_rectangle.msh"

# Petsc -> (0.126s, 0.498s, 54.07s)
# Mumps -> (0.088s, 0.381s, 48.58s)

#File name
filename = os.path.join(MESH_DIR, filename)
   
running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="double", 
                      int_precision="signed")
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
miter = 0
niter = 1
Pinit = 10.
saving_at_node = 1

boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"dirichlet",
              "bottom":"dirichlet",
              }
values = {"in" : 20,
          "out": 0.,
          "upper":0.,
          "bottom":0.,
          }


P  = Variable(domain=domain, BC=boundaries, values=values)
#conf = Struct(reuse_mtx=True, scheme='diamond', verbose=False)
conf = Struct(reuse_mtx=True, scheme='diamond', verbose=False, 
              precond='gamg', sub_precond="amg",# with_mtx=False,
              eps_a=1e-10, eps_r=1e-10, method="gmres")

# L = MUMPSSolver(domain=domain, var=P, conf=conf)
L = PETScKrylovSolver(domain=domain, var=P, conf=conf)
ts = MPI.Wtime()
L()
te = MPI.Wtime()



# P.update_halo_value()
# P.update_ghost_value()
# P.interpolate_celltonode()
#
# domain.save_on_node_multi(0., 0., niter, miter, variables=["P"],values=[P.node])
       

def save_matrix_petsc():
  from scipy.sparse import coo_matrix
  from scipy.io import mmwrite

  # Sparse matrix (COO data you already have)
  row = np.array(L._row)
  col = np.array(L._col)
  data = np.array(L._data)

  print(row)
  # Build sparse matrix
  n = max(max(row), max(col)) + 1
  A = coo_matrix((data, (row, col)), shape=(n, n))
  b = np.array(L.rhs0).reshape(-1, 1)
  x = np.array(L.sol.getArray()).reshape(-1, 1)

  mmwrite("b.mtx", b)
  mmwrite("x.mtx", x)
  mmwrite("A.mtx", A)
  print("Data saved petsc")

def save_matrix_mumps():
  from scipy.sparse import coo_matrix
  from scipy.io import mmwrite

  # Sparse matrix (COO data you already have)
  row = np.array(L._row) - 1
  col = np.array(L._col) - 1
  data = np.array(L._data)

  # Build sparse matrix
  n = max(max(row), max(col)) + 1
  A = coo_matrix((data, (row, col)), shape=(n, n))
  b = np.array(L.rhs0).reshape(-1, 1)
  x = np.array(L.sol).reshape(-1, 1)

  mmwrite("b.mtx", b)
  mmwrite("x.mtx", x)
  mmwrite("A.mtx", A)
  print("Data saved mumps")




tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)
  save_matrix_petsc()
  # save_matrix_mumps()
      