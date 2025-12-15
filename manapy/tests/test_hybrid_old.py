from manapy.tests.GeneralChecker import GeneralChecker
from manapy.tests.helpers.HybridTestTables import HybridTestTables
from manapy.backends.types import FLOAT_TYPE
from manapy.tests.helpers.DomainTables import DomainTables
from manapy.tests.meshes import get_mesh

# Differences between old domain and domain
# halos_int in old domain they are stored using global indexes
# ghost_info reinterpretation of int to float
# Face order inside the cell

def Hybrid():
  test_tables = HybridTestTables(domain_tables.d_cell_loctoglob, dim)

  checker = GeneralChecker(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()

# Create domains
dim, mesh_path, mesh_name = get_mesh(2)
domain_tables = DomainTables(nb_partitions=4, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)
unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)



# Test
Hybrid()




