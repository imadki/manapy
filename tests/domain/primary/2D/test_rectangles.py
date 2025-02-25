from helpers.Checker2D import Checker2D
from helpers.DomainTables import DomainTables

mesh_name = 'rectangles.msh'
float_precision = 'float32'
dim = 2


def test_main():
  partitions = [1, 2, 4, 7] # split the domain into 1, 2, 4, 7 parts for each test
  for nb_partition in partitions:
    domain_tables = DomainTables(nb_partitions=nb_partition, mesh_name=mesh_name, float_precision=float_precision, dim=dim)
    unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim)

    checker = Checker2D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain)
    checker.test_cell_info()
    checker.test_face_info()
    checker.test_node_info()
    checker.test_halo_info()


if __name__ == '__main__':
  test_main()
