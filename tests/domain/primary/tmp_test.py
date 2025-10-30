from mpi4py import MPI
from manapy.partitions import MeshPartition
from manapy.ddm import Domain
from manapy.base.base import Struct
from manapy.domain import Domain as AltDomain
import numpy as np
import os


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]

dim, mesh_name = mesh_list[0]
filename = "/home/aben-ham/Desktop/work/manapy/manapy/tests/meshes/" + mesh_name


running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="double")
mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0, 0, 0])
domain = Domain(dim=dim, conf=running_conf)


local_domain = AltDomain.create_domain(filename, dim, recreate=True)

# if RANK == 0:
# print(RANK, len(domain.cells.nodeid), len(local_domain.cells.nodeid))
if RANK == 0:
  cells = [
    "_nbcells",
    "_nodeid",
    "_faceid",
    "_cellfid",
    "_cellnid",
    "_halonid",
    "_ghostnid",
    "_haloghostnid",
    "_haloghostcenter",
    "_center",
    "_volume",
    "_nf",
    "_loctoglob",
    "_tc",
    "_periodicfid",
    "_shift",
  ]
  nodes = [
    "_nbnodes",
    "_vertex",
    "_name",
    "_oldname",
    "_cellid",
    "_ghostid",
    "_haloghostid",
    "_ghostcenter",
    "_haloghostcenter",
    "_ghostfaceinfo",
    "_haloghostfaceinfo",
    "_loctoglob",
    "_halonid",
    "_nparts",
    "_periodicid",
    "_R_x",
    "_R_y",
    "_R_z",
    "_number",
    "_lambda_x",
    "_lambda_y",
    "_lambda_z",
  ]
  faces = [
    "_nbfaces",
    "_nodeid",
    "_cellid",
    "_name",
    "_oldname",
    "_normal",
    "_mesure",
    "_center",
    "_dist_ortho",
    "_ghostcenter",
    "_oppnodeid",
    "_halofid",
    "_param1",
    "_param2",
    "_param3",
    "_param4",
    "_f_1",
    "_f_2",
    "_f_3",
    "_f_4",
    "_airDiamond",
    "_tangent",
    "_binormal",
  ]
  halos = [
    "_halosext",
    "_neigh",
    "_halosint",
    "_centvol",
    "_sizehaloghost",
    "_faces",
    "_nodes",
  ]
  for item in nodes:
    try:
      s1 = np.array(domain.nodes.__getattribute__(item))
      s2 = np.array(local_domain.nodes.__getattribute__(item))
      if s1.shape != s2.shape:
        print(RANK, item, s1.shape, s2.shape)
    except AttributeError:
      print(RANK, item)


print(local_domain.halos.centvol)
