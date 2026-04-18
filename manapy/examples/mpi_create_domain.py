from manapy.domain import Domain, Partitioning
from manapy.helpers import get_mesh

# TODO fix when MPI change nb_Partitions
dim, mesh_path, mesh_name = get_mesh(2)
local_domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
