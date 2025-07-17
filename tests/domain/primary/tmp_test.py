import subprocess
import os
import numpy as np
import sys

sys.path.append(os.path.join(os.getcwd()))
from tests.domain.primary.helpers.DomainTables import DomainTables

mpi_exec = "/usr/bin/mpirun"
python_exec = "/home/aben-ham/anaconda3/envs/work/bin/python3"
float_precision = 'float32'
dim=3
mesh_name = 'tetrahedron.msh'

def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.getcwd()
  mesh_file_path = os.path.join(root_file, 'mesh', mesh_name)
  script_path = os.path.join(root_file, 'helpers', 'create_partitions_mpi_worker.py')
  cmd = [mpi_exec, "-n", str(nb_partitions), "--oversubscribe", python_exec, script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)

domain_tables = DomainTables(nb_partitions=4, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)


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

checker = TetraChecker3D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain, test_tables=test_tables)
checker.test_cell_info()
checker.test_face_info()
# checker.test_node_info()
# checker.test_halo_info()
checker.summary()
