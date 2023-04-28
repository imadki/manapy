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
running_conf = Struct(backend="numba", signature=True, cache=True, precision="single")
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
cfl = 0.8
time = 0
tfinal = .25
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

ne = Variable(domain=domain)
u  = Variable(domain=domain)
v  = Variable(domain=domain)
w  = Variable(domain=domain)
P  = Variable(domain=domain, BC=boundaries, values=values)

#Call the AdvectionDiff solver
conf = Struct(Dxx=0., Dyy=0., order=2, cfl=0.8)
S = AdvectionDiffusionSolver(ne, (u, v, w), conf=conf)


####Initialisation
initialisation_gaussian_3d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit=Pinit)#, 


f = lambda x, y, z : Pinit * (1. - x)
conf = Struct(reuse_mtx=True, scheme='diamond', verbose=False)
#L = MUMPSSolver(domain=domain, var=P, conf=conf)
L = PETScKrylovSolver(domain=domain, var=P, conf=conf)

ts = MPI.Wtime()

#loop over time
while time < tfinal:
    
    L()
    P.update_halo_value()
    P.update_ghost_value()
    P.interpolate_celltonode()
    L.compute_Sol_gradient()
    
    #TODO -1
    u.face[:] = P.gradfacex[:]
    v.face[:] = P.gradfacey[:]
    w.face[:] = P.gradfacez[:]
    
    u.interpolate_facetocell()
    v.interpolate_facetocell()
    w.interpolate_facetocell()
    
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1

    time = time + d_t
    
    S.compute_fluxes()
    S.compute_new_val()
    
   
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            #save vtk files for the solution
            ne.update_halo_value()
            ne.update_ghost_value()  
            ne.interpolate_celltonode()
            
            #save vtk files for the solution
            u.update_halo_value()
            u.update_ghost_value()  
            u.interpolate_celltonode()
            
            #save vtk files for the solution
            v.update_halo_value()
            v.update_ghost_value()  
            v.interpolate_celltonode()
            
            #save vtk files for the solution
            w.update_halo_value()
            w.update_ghost_value()  
            w.interpolate_celltonode()
            
            domain.save_on_node_multi(d_t, time, niter, miter, variables=["ne", "u","v", "w", "P"],
                                      values=[ne.node, u.node,v.node,w.node, P.node])
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["ne", "u","v", "w","P"],
                                      values=[ne.cell, u.cell,v.cell,w.cell, P.cell])
        miter += 1

    niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)
#L.destroy()
