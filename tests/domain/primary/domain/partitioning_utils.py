from numba.typed import Dict, List
from LocalDomainStructData import new_local_domains
import manapy_domain32
import manapy_domain64


def create_local_domains(part_vert: 'int32[:]', node_cellid: 'int32[:, :]', node_phyid: 'int32[:, :]', cells: 'int32[:, :]', cells_type: 'int8[:]', nodes: 'float64[:, :]', phy_faces: 'int32[:, :]', phy_faces_name: 'int32[:]', nb_parts: 'int32', float_precision: 'int32', dim: 'int32'):

  manapy_domain = manapy_domain64 if float_precision == 64 else manapy_domain32

  print("C -> Create local domains", nodes.dtype)
  c_res = manapy_domain.create_local_domains(part_vert, node_cellid, node_phyid, cells, cells_type, nodes, phy_faces, phy_faces_name, nb_parts)
  print("C -> End Create local domains")

  list_local_domains = new_local_domains(nb_parts)

  for i in range(nb_parts):
    obj = list_local_domains[i]

    k = 0
    obj.nodes = c_res[i][k]; k+=1
    obj.cells = c_res[i][k]; k+=1
    obj.cells_type = c_res[i][k]; k+=1
    obj.phy_faces = c_res[i][k]; k+=1
    obj.phy_faces_name = c_res[i][k]; k+=1
    obj.cell_loctoglob = c_res[i][k]; k+=1
    obj.node_loctoglob = c_res[i][k]; k+=1
    obj.node_oldname = c_res[i][k]; k+=1
    obj.halo_neighsub = c_res[i][k]; k+=1
    obj.node_halos = c_res[i][k]; k+=1
    obj.node_halophyid = c_res[i][k]; k+=1
    obj.phyid_recv = c_res[i][k]; k+=1
    obj.phyid_recv_part_size = c_res[i][k]; k+=1
    obj.phyid_send = c_res[i][k]; k+=1
    obj.halo_halosext = c_res[i][k]; k+=1
    obj.halo_halosint = c_res[i][k]; k+=1
    obj.max_cell_nodeid = c_res[i][k]; k+=1
    obj.max_cell_faceid = c_res[i][k]; k+=1
    obj.max_face_nodeid = c_res[i][k]; k+=1
    obj.max_node_haloid = c_res[i][k]; k+=1
    obj.max_cell_halonid = c_res[i][k]; k+=1
    obj.dim = dim
    obj.float_precision = float_precision

  return list_local_domains
