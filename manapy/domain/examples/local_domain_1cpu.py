import os
from manapy.domain import Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu

# To change float type go to manapy/backends/types.py

mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'big/tetrahedron_big.msh'),
]
root_file = os.getcwd()
dim, mesh_path = mesh_list[4] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', '..', 'tests', 'meshes', mesh_path) # manapy/tests



mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh)
nb_parts = 1000
local_domains = partitioning.create_sub_domains(nb_parts=nb_parts) # intermediate step


print(len(local_domains), local_domains[1].nodes.dtype)


# ld = LocalDomain1Cpu.create_local_domains(local_domains) # list of local domains


