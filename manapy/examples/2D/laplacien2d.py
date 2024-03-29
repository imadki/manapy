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
for test in [2]:#, 2, 4, 8, 16, 32, 64]:
    for scheme in ['diamond']:#, 'fv4']:
        filename = "rectangle_"+str(test)+"K.msh"
        
        if scheme=='fv4':
            with_mtx = True
        else:
            with_mtx = False
        
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
        if RANK == 0: print("Start Computation using",scheme, "in dim", dim, "with mtx", with_mtx)
        miter = 0
        niter = 1
        Pinit = 10.
        saving_at_node = 1
        
        boundaries = {"in" : "dirichlet",
                      "out" : "dirichlet",
                      "upper":"dirichlet",
                      "bottom":"dirichlet",
#                      "front":"dirichlet",
#                      "back":"dirichlet"
                      }
        values = {"in" : 20,
                  "out": 0.,
                  "upper":0.,
                  "bottom":0.,
#                  "front":0.,
#                  "back":0.
                  }
        
        
        P  = Variable(domain=domain, BC=boundaries, values=values)
        
        
        #conf = Struct(reuse_mtx=True, scheme='diamond', verbose=False)
        conf = Struct(reuse_mtx=False, scheme='diamond', verbose=True, 
                          precond='none', #i_max=10000,
                          with_mtx=with_mtx, sub_precond="gamg",
                          eps_a=0, eps_r=1e-14, method="pipefgmres", factor_solver="mumps",
                          reordering=True)
        L = PETScKrylovSolver(domain=domain, var=P, conf=conf)

        import numpy as np
        if scheme== "fv4":
            sizeM = 4*len(L.matrixinnerfaces)+len(L.var.dirichletfaces) + 2*len(L.domain.halofaces)
            L._row  = np.zeros(sizeM, dtype=np.int32)
            L._col  = np.zeros(sizeM, dtype=np.int32)
            L._data = np.zeros(sizeM, dtype=L.float_precision)
            constant = -1
            L.update_ghost_values()
            faces.dist_ortho[:]=1.
            Mat_Assembly(L._row, L._col, L._data,# b,
                         P.ghost, np.ones(nbcells),
                         1., 1., 0,
                         faces.cellid, cells.volume, cells.faceid,
                         faces.mesure, L.matrixinnerfaces,
                         P.dirichletfaces, faces.dist_ortho)
            
            L.rhs0_glob = np.zeros(L.globalsize)
            Vec_Assembly(domain.Pbordface, np.ones(nbcells), 1, 1, 0, faces.cellid,
                         cells.volume, faces.mesure, P.dirichletfaces, P.neumannNHfaces,
                         faces.dist_ortho, L.rhs0, 0, faces.normal)
        
        
        
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
        
        print(" ")
         # Determine the number of rows and columns
        num_rows = max(L._row) + 1
        num_cols = max(L._col) + 1
        
        # Initialize arrays for CSR format
        csr_data = []
        row_ptr = [0] * (num_rows + 1)
        
        # Convert DCR to CSR
        for r, c, val in zip(L._row, L._col, L._data):
            csr_data.append(val)
            row_ptr[r + 1] += 1
        
        # Cumulative sum to get row pointers
        for i in range(1, len(row_ptr)):
            row_ptr[i] += row_ptr[i - 1]
        
        from scipy.sparse import csr_matrix
        from scipy.io import mmwrite
        
        
        # Create a CSR matrix
        A_csr = csr_matrix((csr_data, L._col, row_ptr), shape=(num_rows, num_cols))
#        
        PATH="."#/home/kissami/Documents/GITHUB/ACCELERATION-DOMAIN-DECOMPOSITION-GPU-COMPUTING"
        mmwrite(PATH+'/matrix'+str(test)+'K_'+scheme+'_'+str(dim)+'d.mtx', A_csr)
        np.save(PATH+'/vector'+str(test)+'K_'+scheme+'_'+str(dim)+'d.npy', L.rhs0)
#        np.save(PATH+'/reord'+str(test)+'K_'+scheme+'_'+str(dim)+'d.npy', L.perm)
        
#        del L
#        del L
#        import matplotlib.pyplot as plt
#        
#        fig = plt.figure(figsize=(7,7))
#        plt.spy(A_csr)
#        fig = plt.figure(figsize=(7,7))
#        plt.savefig("Matrix1K.png")
#
        
from scipy.io import mmread
filenameM = 'matrix2K_diamond_2d.mtx'
A = mmread(filenameM).tocsc()
print(A.shape)

#    if len(sys.argv) > 2:
filenameb = 'vector2K_diamond_2d.npy'#sys.argv[2]
b = np.load(filenameb)

#    else:
#        b = np.ones(A.shape[0])
#
#filenamec = 'reord2K_diamond_2d.npy'#sys.argv[2]
#perm = np.load(filenamec)

from scipy.sparse.linalg import gmres
##
#def solve_linear_system(A, b, tol=1e-6, max_iter=100):
#   
#    x = np.zeros_like(b)
#    r = b - A @ x
#
#    num_iter = 0
#    residual_norm = np.linalg.norm(r)
#    while residual_norm > tol and num_iter < max_iter:
#        z, _ = gmres(A, r, x0=np.zeros_like(b),  restart=2)
#        x += z
#        r = b - A @ x
#        residual_norm = np.linalg.norm(r)
#        num_iter += 1
#        print(num_iter, residual_norm)
#    return x, num_iter
##
    
x, _ = gmres(A_csr, b)


##x, num_it = solve_linear_system(A_csr, L.rhs0, tol=1e-8, max_iter=100)
#
#P.cell[:] = x[np.argsort(L.perm)]
##
#P.update_halo_value()
#P.update_ghost_value()
#P.interpolate_celltonode()
#domain.save_on_node_multi(0., 0., niter, miter, variables=["P"],values=[P.node])
##
##
#print("gmres time", time)