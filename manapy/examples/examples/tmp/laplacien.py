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
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver
import os


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

start = timeit.default_timer()

filename = "big/mid_rectangle.msh"
dim, mesh_path, mesh_name = get_mesh(filename)
domain = Domain.create_domain(filename, dim, Partitioning.Par_Nodal, recreate=True)
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


L = MUMPSSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond')

# L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='diamond',
#               precond='gamg', sub_precond="amg",  # with_mtx=False,
#               eps_a=1e-10, eps_r=1e-10, method="gmres")


# Solve the system
ts = MPI.Wtime()
L()
te = MPI.Wtime()



def save_matrix_petsc(x_sol):
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
  x = np.array(x_sol).reshape(-1, 1)

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

def vec_to_numpy_root(x):
  x_local = x.getArray()
  sizes = MPI.COMM_WORLD.allgather(len(x_local))

  if MPI.COMM_WORLD.Get_rank() == 0:
    x_global = np.empty(sum(sizes))
  else:
    x_global = None

  MPI.COMM_WORLD.Gatherv(
    x_local,
    (x_global, sizes) if MPI.COMM_WORLD.Get_rank() == 0 else None,
    root=0
  )
  if MPI.COMM_WORLD.Get_rank() == 0:
    return x_global
  return None


tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

if isinstance(L, PETScKrylovSolver):
  x_sol = vec_to_numpy_root(L.sol)
  if RANK == 0:
    save_matrix_petsc(x_sol)
else:
  if RANK == 0:
    save_matrix_mumps()
