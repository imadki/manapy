import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'domain'))
from domain.create_domain import Domain, Mesh, GlobalDomain, LocalDomain, Partitioning
from domain.local_domain_1cpu_testing import LocalDomain1Cpu, SingleCoreDomainTables
import numpy as np


def Cube():
  # Cube
  import importlib

  module = importlib.import_module("helpers.TablesTestHexa3D")
  importlib.reload(module)
  TestTablesRect2D = getattr(module, "TablesTestHexa3D")

  module = importlib.import_module("helpers.Checker3D")
  importlib.reload(module)
  Checker3D = getattr(module, "Checker3D")

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TestTablesRect2D(float_precision, d_cell_loctoglob, g_cell_nodeid)
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
  import importlib

  module = importlib.import_module("helpers.TablesTestRect2D")
  importlib.reload(module)
  TestTablesRect2D = getattr(module, "TablesTestRect2D")

  module = importlib.import_module("helpers.Checker2D")
  importlib.reload(module)
  Checker2D = getattr(module, "Checker2D")

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TestTablesRect2D(float_precision, d_cell_loctoglob, g_cell_nodeid)
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
  import importlib

  module = importlib.import_module("helpers.TablesTestTriangles2D")
  importlib.reload(module)
  TestTablesTriangles2D = getattr(module, "TablesTestTriangles2D")

  module = importlib.import_module("helpers.Checker2D")
  importlib.reload(module)
  Checker2D = getattr(module, "Checker2D")

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TestTablesTriangles2D(float_precision, d_cell_loctoglob, g_cell_nodeid)
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
  import importlib

  module = importlib.import_module("helpers.TablesTestTetra3D")
  importlib.reload(module)
  TablesTestTetra3D = getattr(module, "TablesTestTetra3D")

  module = importlib.import_module("helpers.TetraChecker3D")
  importlib.reload(module)
  TetraChecker3D = getattr(module, "TetraChecker3D")

  d_cell_loctoglob = domain_tables.d_cell_loctoglob
  g_cell_nodeid = unified_domain.d_cell_nodeid[0]
  test_tables = TablesTestTetra3D(float_precision, d_cell_loctoglob, g_cell_nodeid)
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
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path, test_function = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'mesh', mesh_path) #tests/domain/primary/mesh

def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh, float_precision)
  local_domain_data = partitioning.create_sub_domains(nb_parts=nb_parts)

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, SingleCoreDomainTables(domains, float_precision)

l_domains, domain_tables = create_domain(100)
g_domains, unified_domain = create_domain(1)


test_function()