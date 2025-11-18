from manapy.domain import Domain
from manapy.tests.meshes import get_mesh

dim, mesh_path, mesh_name = get_mesh(3)
local_domain = Domain.create_domain(mesh_path, dim, recreate=True)
