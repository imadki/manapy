import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'domain'))
from create_domain import Mesh, Partitioning, LocalDomain
from local_domain_1cpu_testing import LocalDomain1Cpu

mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float64' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[0] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh



mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh, float_precision)
nb_parts = 1
local_domains = partitioning.create_sub_domains(nb_parts=nb_parts)

ld = LocalDomain1Cpu.create_local_domains(local_domains)


