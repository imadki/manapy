#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:59:14 2026

@author: kissami
"""

import os
import numpy as np
from mpi4py import MPI
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.ls import GinkgoSolver, MUMPSSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

MESH_DIR = "/home/kissami/Documents/GITHUB/manapy/meshes/geo"
mesh_path = os.path.join(MESH_DIR, "carre.msh")
domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)

boundaries = {"in": "dirichlet", "out": "dirichlet", "upper": "dirichlet", "bottom": "dirichlet"}
values = {"in": 20, "out": 0., "upper": 0., "bottom": 0.}

# Reference: MUMPS (parallel, centralized)                                                                                                                             
Pm = Variable(domain=domain, BC=boundaries, values_dict=values)
Lm = MUMPSSolver(domain=domain, var=Pm, reuse_mtx=True, scheme='diamond')
Lm()
cell_ref = Pm.cell.copy()

# Ginkgo on CPU                                                                                                                                                        
Pg = Variable(domain=domain, BC=boundaries, values_dict=values)
Lg = GinkgoSolver(domain=domain, var=Pg, reuse_mtx=True, scheme='diamond',
                  device="cpu", method="gmres", precond="ilu", eps_r=1e-12, i_max=2000)
Lg()
cell_g = Pg.cell.copy()

# Compare local var.cell on each rank (distributed solution)                                                                                                           
loc = np.linalg.norm(cell_g - cell_ref) / (np.linalg.norm(cell_ref) + 1e-30)
glob = COMM.allreduce(loc, op=MPI.MAX)
if RANK == 0:
    print(f"[parallel n={COMM.Get_size()}] max rel error Ginkgo vs MUMPS (var.cell) = {glob:.3e}")