from manapy.domain import Domain, Partitioning
from manapy.tests.meshes import get_mesh

dim, mesh_path, mesh_name = get_mesh(1)
local_domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
