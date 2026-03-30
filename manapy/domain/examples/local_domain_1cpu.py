from manapy.domain import Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu
from manapy.tests.meshes import get_mesh

# To change float type go to manapy/backends/types.py
dim, mesh_path, mesh_name = get_mesh(1)
mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh)
nb_parts = 4
partitioning.make_n_part_mesh_nodal(nb_parts)
local_domains = partitioning.create_sub_domains() # intermediate step


ld = LocalDomain1Cpu.create_local_domains(local_domains) # list of local domains

