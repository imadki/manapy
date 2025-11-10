import os
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.tests.GeneralChecker import GeneralChecker
from manapy.tests.helpers.GeneralTestTables import GeneralTestTables
from manapy.tests.HybridChecker2D import HybridChecker2D
from manapy.tests.helpers.HybridTestTables import HybridTestTables
from manapy.backends.types import FLOAT_TYPE

def General():
  cell_loctoglob = domain_tables.d_cell_loctoglob
  cells = unified_domain.d_cell_nodeid[0]
  nodes = unified_domain.d_nodes[0]
  phy_faces = global_domain[0].phy_faces
  phy_faces_name = global_domain[0].phy_faces_name
  cell_type = global_domain[0].cells_type
  max_cell_faceid = global_domain[0].max_cell_faceid
  max_face_nodeid = global_domain[0].max_face_nodeid
  test_tables = GeneralTestTables(cell_loctoglob, cells, nodes, phy_faces, phy_faces_name, cell_type, max_cell_faceid, max_face_nodeid, dim)

  checker = GeneralChecker(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()

def Hybrid():
  cell_loctoglob = domain_tables.d_cell_loctoglob
  test_tables = HybridTestTables(cell_loctoglob, dim)

  checker = GeneralChecker(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  # checker.test_face_info()
  # checker.test_node_info()
  # checker.test_halo_info()
  checker.summary()

mesh_list = [
  (2, 'rectangles.msh', General),
  (2, 'triangles.msh', General),
  (2, 'hybrid.msh', Hybrid),
  (3, 'cube.msh', General),
  (3, 'tetrahedron.msh', General),
  (3, 'tetrahedron_big.msh', General),
]
root_file = os.getcwd()
dim, mesh_path, test_function = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'meshes', mesh_path) #tests/domain/primary/mesh

def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  local_domain_data = partitioning.create_sub_domains(nb_parts=nb_parts)

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, SingleCoreDomainTables(domains, FLOAT_TYPE), local_domains

l_domains, domain_tables, local_domain = create_domain(4)
g_domains, unified_domain, global_domain = create_domain(1)







test_function()




