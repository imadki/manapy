```
from manapy.domain import Domain, Partitioning
from manapy.tests.meshes import get_mesh

dim, mesh_path, mesh_name = get_mesh(1)
local_domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)

MeshClass (rank0)
   │
   ├── generates → Mesh (cells, nodes, physical faces, ...)
   │
   ▼
PartitioningClass (rank0)
   │
   ├── takes → Mesh
   ├── produces → List[LocalDomainInterface] (create_sub_domains)
   │
   ▼
LocalDomainInterface (rank0)
   │── (start from this step if recreate=False)
   ├── acts as → input for LocalDomainClass
   │
   ▼
LocalDomainClass (for each rank)
   │
   ├── processes → LocalDomainInterface
   ├── produces → LocalDomain
   │
   ▼
DomainClass (for each rank)
   │
   ├── takes → LocalDomain
   └── builds → Domain for each rank
```