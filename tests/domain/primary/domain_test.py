import os
import numpy as np
from create_domain import Domain, Mesh, LocalDomain
import gc

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
local_domains_data = domain.c_create_sub_domains(4) # Number of partitions
print("====> End <=====")

print(local_domains_data[0].max_cell_faceid)
local_domain = LocalDomain(local_domains_data[0])

# # Release Memory
# mesh = None
# domain = None
# local_domains_data = None
# gc.collect()
#
#
# print("====> Start <=====")
# # Number of partitions is determined on meshpartitioning.py:121 => self._size = 3000
# from manapy.partitions import MeshPartition
# from manapy.base.base import Struct
# import time
#
# start = time.time()
# running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
# a = MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
# print(f"Execution time: {time.time() - start:.6f} seconds")
# print("====> End <=====")