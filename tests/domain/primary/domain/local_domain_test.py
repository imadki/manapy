import os
import numpy as np
from create_domain import Domain, Mesh, GlobalDomain, LocalDomain, log_step
import time
from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

mesh_list = [
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh


# ------------------------------------------------------------------
# 1. Start
# ------------------------------------------------------------------
mesh = Mesh(mesh_path, dim)
domain = GlobalDomain(mesh, float_precision)
local_domain_data = domain.c_create_sub_domains(4)

# print(local_domain_data[0].node_oldname)

local_domains = LocalDomain.create_local_domains(local_domain_data)

log_step.print_resutls()

