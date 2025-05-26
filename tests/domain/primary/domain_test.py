import os
import numpy as np
from create_domain import Domain, Mesh
import gc


float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = (3, 'tetrahedron_big.msh') # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'mesh', mesh_path) #tests/domain/primary/mesh


# Fast and uses less ram
print("====> Start <=====")
mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
print(domain.cells.shape[0])
local_domains_data = domain.c_create_sub_domains(3000) # Number of partitions
print("====> End <=====")

# Release Memory
mesh = None
domain = None
local_domains_data = None
gc.collect()


print("====> Start <=====")
# Number of partitions is determined on meshpartitioning.py:121 => self._size = 3000
from manapy.partitions import MeshPartition
from manapy.base.base import Struct
import time

start = time.time()
running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
a = MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
print(f"Execution time: {time.time() - start:.6f} seconds")
print("====> End <=====")