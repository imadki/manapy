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

from manapy.solvers.advecdiff.tools_utils import initialisation_gaussian_2d
from manapy.solvers.advecdiff import AdvectionDiffusionSolver

from manapy.ast import Variable
from manapy.base.base import Struct

import os

from tool import *

if RANK == 0:
    print(RANK, SIZE)

import os

start = timeit.default_timer()
# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..','..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')

import sys;
filename = "carre_hybrid.msh"#sys.argv[1]
backend = "numba"#sys.argv[2]

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 2

###Config###
#backend numba or python
#signature: add types to functions (make them faster) but compilation take time
#cache: avoid recompilation in the next run
running_conf = Struct(backend=backend, signature=True, cache=True)#, precision="double")
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


ts = MPI.Wtime()
# Identifier le maître par nœud (par exemple, rangs divisibles par le nombre de processus par nœud)
num_processes_per_node = 56  
is_node_master = (RANK % num_processes_per_node == 0)


if is_node_master:
    # Initial energy readings
    initial_energy_socket0 = read_energy(0)
    initial_energy_socket1 = read_energy(1)
else:
    initial_energy_socket0 = 0
    initial_energy_socket1 = 0

# Initialize pyRAPL for energy measurement
pyRAPL.setup()
meter = pyRAPL.Measurement('bar')
meter.begin()

#TODO tfinal
if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .15
miter = 0
niter = 1
Pinit = 2.
saving_at_node = 1

boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u  = Variable(domain=domain)
v  = Variable(domain=domain)
P  = Variable(domain=domain, BC=boundaries, values=values)

#Call the transport solver
conf = Struct(Dxx=0.01, Dyy=0., order=2, cfl=0.8)
S = AdvectionDiffusionSolver(ne, vel=(u, v), conf=conf)

####Initialisation
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z : Pinit * (1. - x)

if RANK == 0: print("Start While loop ...")

freq = {}
#loop over time
while time < tfinal:
    
    #TODO -1
    u.face[:] = 2.
    v.face[:] = 0.
    
    u.interpolate_facetocell()
    v.interpolate_facetocell()

    d_t = S.stepper()
    tot = int(tfinal/d_t/10)+1
    
    time = time + d_t
    
    S.compute_fluxes()
    S.compute_new_val()

    if niter== 1 or niter%tot == 0:
            
        freq_after = get_cpu_frequency()
        freq[niter] = COMM.allreduce(freq_after[global_to_local_index(RANK)]/SIZE, op=MPI.SUM)
    
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
   
    
            #domain.save_on_node_multi(d_t, time, niter, miter, variables=["ne", "u","v", "P"],
            #                          values=[ne.node, u.node,v.node, P.node], file_format="vtu")
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["ne", "u","v","P"],
                                      values=[ne.cell, u.cell,v.cell, P.cell], file_format="vtu")
        miter += 1

    niter += 1

te = MPI.Wtime()


frequencies_after = get_cpu_frequency()

meter.end()
# End measuring energy after the function execution
tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)

if is_node_master:
    # Final energy readings
    final_energy_socket0 = read_energy(0)
    final_energy_socket1 = read_energy(1)
else:
    # Final energy readings
    final_energy_socket0 = 0
    final_energy_socket1 = 0
    
# Calculate energy consumed in microjoules
energy_socket0 = final_energy_socket0 - initial_energy_socket0
energy_socket1 = final_energy_socket1 - initial_energy_socket1

energy_consumed = energy_socket0 + energy_socket1
total_energy_1 = COMM.reduce(energy_consumed * 1e-6, op=MPI.SUM, root=0)

if RANK == 0:
    print(f"Energy consumed: {total_energy_1:.4f} Joules")

    
## Get energy consumption
energy_consumed = 0
if is_node_master:
    if meter.result.pkg is not None:
        energy_consumed_socket0 = meter.result.pkg[0] if meter.result.pkg[0] is not None else 0
        energy_consumed_socket1 = meter.result.pkg[1] if meter.result.pkg[1] is not None else 0
        energy_consumed = energy_consumed_socket0 + energy_consumed_socket1
        #print(energy_consumed * 1e-6, RANK, socket.gethostname())
        
total_energy_2 = COMM.reduce(energy_consumed * 1e-6, op=MPI.SUM, root=0)

if RANK == 0:
    print(f"Energy consumed: {total_energy_2:.4f} Joules")
    print("cpu time using",size,"is ", tt)
    print("frequencies", freq)
    #power = total_energy / tt
    #print(f"Power consumed: {power:.4f} Watts")
    mesh_dir = "meshes"+str(SIZE)+"PROC"
    os.system("rm -fr "+mesh_dir)
    os.system("rm -fr results")
