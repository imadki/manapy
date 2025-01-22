import sys
from manapy.partitions import MeshPartition
from manapy.base.base import Struct
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def create_partitions(mesh_file_path, float_precision, dim):
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
  MeshPartition(mesh_file_path, dim=dim, conf=running_conf, periodic=[0,0,0])


if len(sys.argv) == 4:
  mesh_file_path = sys.argv[1]
  float_precision = sys.argv[2]
  dim = int(sys.argv[3])
  if (float_precision == 'float32' or float_precision == 'float64') and (dim == 2 or dim == 2):
    print("path", mesh_file_path, "precision", float_precision, "dim", dim, "rank", rank)
    create_partitions(mesh_file_path, float_precision, dim)
  else:
    raise Exception("Invalid float_precision argument or Invalid dim argument")
else:
  raise Exception("Invalid mesh_file_path argument")

