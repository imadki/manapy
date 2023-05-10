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
from manapy.solvers.ls import PETScKrylovSolver, MUMPSSolver, ScipySolver
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
 
#TODO tfinal
if RANK == 0: print("Start Computation ...")
Pinit = 10.


ts = MPI.Wtime()

def test2d_1():
    
    filename = "carre_hybrid.msh"
    
    #File name
    filename = os.path.join(MESH_DIR, filename)
    dim = 2
    
    ###Config###
    #backend numba or python
    #signature: add types to functions (make them faster) but compilation take time
    #cache: avoid recompilation in the next run
    running_conf = Struct(backend="numba", signature=True, cache=True, precision="single")
    mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim, conf=running_conf)
    cells = domain.cells
    nbcells = domain.nbcells

    
    boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
    values = {"in" : Pinit,
              "out": 0.,
              }
    
    w  = Variable(domain=domain, BC=boundaries, values=values)
    
    ###Linear sys confi###
    #If you want the default options please do conf = Struct()
    #reuse_mtx: matrix does not change during the while loop
    #scheme: diamond (fv4 not tested!!!)
    #verbose: printing the mumps/petsc output
    conf = Struct(reuse_mtx=False, scheme='diamond', verbose=False, 
                  precond='gamg', sub_precond="amg",
                  eps_a=1e-10, eps_r=1e-10, method="gmres")
    L = PETScKrylovSolver(domain=domain, var=w, conf=conf)

    
    ####Initialisation
    f = lambda x, y, z : Pinit * (1. - x)

    if RANK == 0:
        rhs = np.zeros(L.globalsize)
        rhs[:] = 0.
    else:
        rhs = None
        
    L(rhs=rhs)
    
    w.update_halo_value()
    w.update_ghost_value()
    # interpolate value on node
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,1,value=w.node) 
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)

def test2d_2():
    
    filename = "carre.msh"
    
    #File name
    filename = os.path.join(MESH_DIR, filename)
    dim = 2
    
    ###Config###
    #backend numba or python
    #signature: add types to functions (make them faster) but compilation take time
    #cache: avoid recompilation in the next run
    running_conf = Struct(backend="numba", signature=True, cache=True, precision="single")
    mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim, conf=running_conf)
    cells = domain.cells
    nbcells = domain.nbcells
    
    f = lambda x, y, z : x*(x-1) + y*(y-1)
      
    boundaries = {"in" : "dirichlet",
                  "out" : "dirichlet",
                  "upper":"dirichlet",
                  "bottom":"dirichlet"
                  }
    values = {"in" : f,
              "out": f,
              "upper":f,
              "bottom":f
              }
    
    w = Variable(domain=domain, BC=boundaries, values=values)
    
    ###Linear sys confi###
    #If you want the default options please do conf = Struct()
    #reuse_mtx: matrix does not change during the while loop
    #scheme: diamond (fv4 not tested!!!)
    #verbose: printing the mumps/petsc output
    conf = Struct(reuse_mtx=False, scheme='diamond', verbose=False)
    L = MUMPSSolver(domain=domain, var=w, conf=conf)
    
    if RANK == 0:
        rhs = np.zeros(L.globalsize)
        rhs[:] = 4.
    else:
        rhs = None
        
    L(rhs=rhs)
    
    # interpolate value on node
    w.update_halo_value()
    w.update_ghost_value()
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,2,value=w.node) 
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)


test2d_1()
#test2d_2()
