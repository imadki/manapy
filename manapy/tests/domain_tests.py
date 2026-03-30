import os
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.tests import Checker2D, Checker3D, TetraChecker3D, TablesTestRect2D, TablesTestHexa3D, TablesTestTetra3D, TablesTestTriangles2D, DomainTables
import manapy.backends.types as types



def Cube():
  # Cube


  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestHexa3D(d_cell_loctoglob, g_cell_nodeid)
  test_tables.init()

  checker = Checker3D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()


def Rectangle():
  # Rectangle
  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestRect2D(d_cell_loctoglob, g_cell_nodeid)
  test_tables.init()

  checker = Checker2D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()


def Triangle():
  # Triangle

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestTriangles2D(d_cell_loctoglob, g_cell_nodeid)
  test_tables.init()

  checker = Checker2D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()


def Tetra():
  # Tetra
  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestTetra3D(d_cell_loctoglob, g_cell_nodeid)
  test_tables.init()

  checker = TetraChecker3D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                           test_tables=test_tables)
  checker.test_cell_info()
  checker.test_node_info()
  checker.test_face_info()
  checker.test_halo_info()
  checker.summary()



mesh_list = [
  (2, 'rectangles.msh', Rectangle),
  (2, 'triangles.msh', Triangle),
  (3, 'cube.msh', Cube),
  (3, 'tetrahedron.msh', Tetra),
  (3, 'tetrahedron_big.msh', None),
]
root_file = os.getcwd()
dim, mesh_name, test_function = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'meshes', mesh_name) #tests/domain/primary/mesh

def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  partitioning.make_n_part_mesh_nodal(nb_parts)
  local_domain_data = partitioning.create_sub_domains()

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, SingleCoreDomainTables(domains)

mpi_test = False
if not mpi_test:
  l_domains, domain_tables = create_domain(12)
  g_domains, unified_domain = create_domain(1)
else:
  domain_tables  = DomainTables(4, mesh_name, dim, None)
  unified_domain = DomainTables(1, mesh_name, dim, None)

test_function()