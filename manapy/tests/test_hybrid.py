import os
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.tests.GeneralChecker import GeneralChecker
from manapy.tests.helpers.GeneralTestTables import GeneralTestTables
from manapy.tests.helpers.HybridTestTables import HybridTestTables
from manapy.backends.types import FLOAT_TYPE
import numpy as np
from manapy.tests.meshes import get_mesh

def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  if nb_parts > 1:
    partitioning.make_n_part_mesh_nodal(nb_parts)
  local_domain_data = partitioning.create_sub_domains()

  if nb_parts == 1:
    local_domain_data[0].cell_loctoglob = np.arange(len(partitioning.cells), dtype=np.int32)
    local_domain_data[0].node_loctoglob = np.arange(len(partitioning.nodes), dtype=np.int32)

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  part_vert = partitioning.part_vert
  if part_vert is None:
    part_vert = np.zeros(len(partitioning.cells), dtype=np.int32)

  return domains, SingleCoreDomainTables(domains, FLOAT_TYPE), local_domains, part_vert

dim, mesh_path, mesh_name = get_mesh(2)
l_domains, domain_tables, local_domain, g_part_vert = create_domain(4)
g_domains, unified_domain, global_domain, _ = create_domain(1)

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
  test_tables = HybridTestTables(domain_tables.d_cell_loctoglob, dim)

  checker = GeneralChecker(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain,
                      test_tables=test_tables)
  checker.test_cell_info()
  checker.test_face_info()
  checker.test_node_info()
  checker.test_halo_info()
  checker.summary()



General()




