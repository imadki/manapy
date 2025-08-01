import os
import numpy as np
from create_domain import Domain, Mesh, GlobalDomain, LocalDomain, log_step
import time
from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

mesh_list = [
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh


# ------------------------------------------------------------------
# 1. Start
# ------------------------------------------------------------------
import manapy_domain32


mesh = Mesh(mesh_path, dim)
domain = GlobalDomain(mesh, float_precision)

log_step("node_cellid")
node_cellid = domain.create_node_cellid(domain.cells, domain.nb_nodes)
log_step()

log_step("cell_cellnid")
cell_cellnid = domain.create_cell_cellnid(domain.cells, node_cellid)
log_step()

log_step("cell_cellfid and boundary_faces")
(
  cell_cellfid,
  bf_cellid,
  bf_nodes
) = domain._create_cellfid_bf_info(domain.cells, node_cellid, domain.cells_type, domain.max_cell_faceid,
                                 domain.max_face_nodeid, domain.nb_phy_faces)
log_step()

log_step("make_n_part")
ret = manapy_domain32.make_n_part(cell_cellfid, 4)
print()
log_step()


print("------")

log_step("make_n_part_mesh_dual")
ret = manapy_domain32.make_n_part_mesh_dual(domain.cells, len(domain.nodes), 4, 3)
print()
log_step()
# print(ret)
# local_domain_data = domain.c_create_sub_domains(4)

# print(local_domain_data[0].node_oldname)

# print(local_domain_data[0].nodes.dtype)
# print(local_domain_data[0].node_halos.dtype)

# local_domains = LocalDomain.create_local_domains(local_domain_data)


# log_step.print_resutls()

