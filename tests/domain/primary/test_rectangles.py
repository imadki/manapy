from helpers.Checker2D import Checker2D
from helpers.DomainTables import DomainTables
from helpers.TablesTestRect2D import TablesTestRect2D
import pytest

mesh_name = 'rectangles.msh'
float_precision = 'float32'
dim = 2

test_cases = [ # split the domain into 1, 2, 4, 7 parts for each test
  (1),
  (2),
  (4),
  (7)
]
@pytest.mark.parametrize("nb_partition", test_cases)
def test_main(nb_partition):
  domain_tables = DomainTables(nb_partitions=nb_partition, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
  unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim)

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestRect2D(float_precision, d_cell_loctoglob, g_cell_nodeid)
  test_tables.init()

  checker = Checker2D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain, test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  #checker.summary()
  assert checker.summary() == True


if __name__ == '__main__':
  test_main()