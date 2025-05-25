import os
import numpy as np
from create_domain import Domain, Mesh
import gc

dim = 3
float_precision = 'float32'
root_file = os.getcwd()
mesh_path = 'tetrahedron.msh'
mesh_path = os.path.join(root_file, 'mesh', mesh_path)


# Fast and uses less ram
print("====> Start <=====")
mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
print(domain.cells.shape[0])
local_domains_data = domain.c_create_sub_domains(4) # Number of partitions
print("====> End <=====")

# Release Memory
mesh = None
domain = None
local_domains_data = None
gc.collect()


print("====> Start <=====")
# Number of partitions is determined on meshpartitioning.py:121 => self._size = 4
from manapy.partitions import MeshPartition
from manapy.base.base import Struct
import time

start = time.time()
running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
a = MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
print(f"Execution time: {time.time() - start:.6f} seconds")
print("====> End <=====")