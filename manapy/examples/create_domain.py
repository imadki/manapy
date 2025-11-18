from mpi4py import MPI
from manapy.partitions import MeshPartition
from manapy.ddm import Domain
from manapy.base.base import Struct
import traceback
from manapy.tests.meshes import get_mesh


try:
  dim, mesh_path, mesh_name = get_mesh("rectangles.msh")
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="double")
  mesh = MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])
  domain = Domain(dim=dim, conf=running_conf)
except Exception as e:
  print("Error: ", traceback.format_exc())
  MPI.COMM_WORLD.Abort(1)