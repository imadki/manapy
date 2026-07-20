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
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells


ts = MPI.Wtime()

if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .15
miter = 0
niter = 1
Pinit = 2.
saving_at_node = 1

ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
w = Variable(domain=domain)
P = Variable(domain=domain)

S = AdvectionDiffusionSolver(ne, vel=(u, v, w), Dxx=0.01, Dyy=0., order=2, cfl=0.8)

initialisation_gaussian_3d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)

if RANK == 0: print("Start While loop ...")

while time < tfinal:

  u.face[:] = 2.
  v.face[:] = 0.
  w.face[:] = 0.

  u.interpolate_facetocell()
  v.interpolate_facetocell()
  w.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 10) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      ne.update_halo_value()
      ne.update_ghost_value()
      ne.interpolate_celltonode()

      u.update_halo_value()
      u.update_ghost_value()
      u.interpolate_celltonode()

      v.update_halo_value()
      v.update_ghost_value()
      v.interpolate_celltonode()

      w.update_halo_value()
      w.update_ghost_value()
      w.interpolate_celltonode()

      domain.save_on_node_multi(["ne", "u", "v", "P"],
                                [ne.node, u.node, v.node, P.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "u", "v", "P"], [ne.cell, u.cell, v.cell, P.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

