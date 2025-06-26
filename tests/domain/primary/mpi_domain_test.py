import os
import numpy as np
from create_domain import Domain, Mesh, LocalDomain
import gc

from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

mesh_list = [
  (2, 'triangles.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[1] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'mesh', mesh_path) #tests/domain/primary/mesh

# from manapy.partitions import MeshPartition
# from manapy.base.base import Struct
# from manapy.ddm import Domain
#
# running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
# a = MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
# domain = Domain(dim=dim, conf=running_conf)

# Fast and uses less ram
print("====> Start <=====")
mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
print(domain.cells.shape[0])
local_domains_data = domain.c_create_sub_domains(size) # Number of partitions
print(f"====> End Rank = {rank} <=====")

print(f"====> LocalDomain {rank} <=====")
local_domain = LocalDomain(local_domains_data[rank], rank)


