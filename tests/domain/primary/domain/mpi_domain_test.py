import os
import numpy as np
from create_domain import Domain

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
dim, mesh_path = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh


def create_original_domain(recreate=True):
  from manapy.partitions import MeshPartition
  from manapy.base.base import Struct
  from manapy.ddm import Domain

  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  if recreate:
    MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])

  return Domain(dim=dim, conf=running_conf)

import time
if rank == 0:
  start = time.time()

domain = Domain.create_domain(mesh_path, dim, float_precision, recreate=False)
# domain = create_original_domain(recreate=False)

MPI.COMM_WORLD.Barrier()
if rank == 0:
  print(f"END:: Execution time: {time.time() - start:.6f} seconds")

"""
Tetra 6000 4
Ori Domain: 0.61s || 0.28s
Alt Domain: 0.30s || 0.20s

Tetra 6000000 4
Ori Domain: 250s Ram Pick 21GB || 167s
Alt Domain: 43s Ram Pick 13GB || 14s

"""
