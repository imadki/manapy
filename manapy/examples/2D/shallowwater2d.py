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

from manapy.solvers.shallowater import ShallowWaterSolver
from manapy.solvers.shallowater.tools_utils import initialisation_SW

from manapy.ast import Variable
from manapy.base.base import Struct

import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..','..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "rectangle_2K.msh"

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 2

###Config###
#backend numba or python
#signature: add types to functions (make them faster) but compilation take time
#cache: avoid recompilation in the next run
running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="double")#, int_precision="signed")
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
if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .06
miter = 0
niter = 1
saving_at_node = 1

boundaries = {"in" : "neumann",
              "out" : "neumann",
              "upper":"nonslip",
              "bottom":"nonslip"
              }

h   = Variable(domain=domain)
hu  = Variable(domain=domain, BC=boundaries)
hv  = Variable(domain=domain, BC=boundaries)
hc  = Variable(domain=domain)
Z   = Variable(domain=domain)

initialisation_SW(h.cell, hu.cell, hv.cell, hc.cell, Z.cell, cells.center)

#Call the transport solver
conf = Struct(order=2, cfl=0.8)
S = ShallowWaterSolver(h=h,  hvel=(hu, hv), hc=hc, Z=Z, conf=conf)


ts = MPI.Wtime()
if RANK == 0: print("Start While loop ...")

#loop over time
while time < tfinal:
    
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1

    time = time + d_t
    
    S.compute_fluxes()
    S.compute_new_val()
    
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            #save vtk files for the solution
            h.update_halo_value()
            h.update_ghost_value()  
            h.interpolate_celltonode()
            
            #save vtk files for the solution
            hu.update_halo_value()
            hu.update_ghost_value()  
            hu.interpolate_celltonode()
            
            #save vtk files for the solution
            hv.update_halo_value()
            hv.update_ghost_value()  
            hv.interpolate_celltonode()
   
    
            domain.save_on_node_multi(d_t, time, niter, miter, variables=["h", "hu","hv"],
                                      values=[h.node, hu.node, hv.node])
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["h", "hu","hv"],
                                      values=[h.cell, hu.cell, hv.cell])
        miter += 1

    niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)

