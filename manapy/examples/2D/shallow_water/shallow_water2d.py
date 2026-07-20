#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
from manapy.solvers.shallowater.system import ShallowWaterSolver
from manapy.solvers.shallowater.tools_utils_compute import initialisation_SW
import timeit
import os
from manapy.domain import Domain, Partitioning
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
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = 'uns_square.msh'
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

# TODO tfinal
if RANK == 0:
  print("Start Computation ...")

time = 0
tfinal = .06
miter = 0
niter = 1
saving_at_node = 1

boundaries = {"in": "neumann",
              "out": "neumann",
              "upper": "slip",
              "bottom": "slip"
              }

h = Variable(domain=domain)
hu = Variable(domain=domain, BC=boundaries)
hv = Variable(domain=domain, BC=boundaries)
hc = Variable(domain=domain)
Z = Variable(domain=domain)

initialisation_SW(h.cell, hu.cell, hv.cell, hc.cell, Z.cell, cells.center, 0)

S = ShallowWaterSolver(h=h, hvel=(hu, hv), hc=hc, Z=Z, order=1, cfl=0.8)

ts = MPI.Wtime()
if RANK == 0:
  print("Start While loop ...")

while time < tfinal:

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      h.update_halo_value()
      h.update_ghost_value()
      h.interpolate_celltonode()

      hu.update_halo_value()
      hu.update_ghost_value()
      hu.interpolate_celltonode()

      hv.update_halo_value()
      hv.update_ghost_value()
      hv.interpolate_celltonode()

      domain.save_on_node_multi(["h", "hu", "hv"],
                                [h.node, hu.node, hv.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["h", "hu", "hv"],
                                [h.cell, hu.cell, hv.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)

