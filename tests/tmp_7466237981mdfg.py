
import pickle
from manapy.domain import Domain, Partitioning, Mesh
from manapy.backends.config import ManapyConfig
from manapy.helpers import get_test_mesh
from mpi4py import MPI
import os
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
config = ManapyConfig(float_precision='float64',
                      int_precision='int64',
                      device='cpu',
                      verbose=False)
local_domain = Domain.create_domain('/run/media/aben-ham/SSD/aben-ham/work/new_manapy/manapy/helpers/../../tests/data/meshes/hybrid2d.msh', 2, config, Partitioning.Par_Nodal, recreate=True)

mesh = Mesh('/run/media/aben-ham/SSD/aben-ham/work/new_manapy/manapy/helpers/../../tests/data/meshes/hybrid2d.msh', 2, config)
partitioner = Partitioning(mesh)
if size > 1:
  part_vert, nb_parts = partitioner.set_part_vert(size, Partitioning.Par_Nodal)
else:
  part_vert = np.zeros(shape=(len(mesh.cells)), dtype=np.int32)

mesh_dir = "domain_meshes" + str(size) + "PROC"
if rank == 0 and not os.path.exists(mesh_dir):
  os.mkdir(mesh_dir)

MPI.COMM_WORLD.barrier()

filename = os.path.join(mesh_dir, f"domain_{rank}.pkl")
print(f"saving mesh {filename}")
if os.path.exists(filename):
  os.remove(filename)

def make_picklable(obj):
  for k, v in vars(obj).items():
    try:
      pickle.dumps(v)
    except:
      setattr(obj, k, None)
  return obj

with open(filename, "wb") as f:
  pickle.dump(make_picklable(local_domain), f)

if rank == 0:
  filename = os.path.join(mesh_dir, '..', f"part_vert.pkl")
  with open(filename, "wb") as f:
    pickle.dump(part_vert, f)
  