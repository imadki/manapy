import pickle
from manapy.domain import Domain, Partitioning, Mesh
from manapy.tests.meshes import get_mesh
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dim, mesh_path, mesh_name = get_mesh(1)
local_domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)

mesh = Mesh(mesh_path, dim)
partitioner = Partitioning(mesh)
part_vert, nb_parts = partitioner.set_part_vert(size, Partitioning.Par_Nodal)

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
  filename = os.path.join(mesh_dir, f"part_vert.pkl")
  with open(filename, "wb") as f:
    pickle.dump(part_vert, f)