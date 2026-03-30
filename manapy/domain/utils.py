import numpy as np
from manapy.domain import domain_compute
import manapy.backends.types as types

# LocalDomainClass.py and Partitioning.py
def create_node_cellid(cells: 'int[:, :]', nb_nodes: 'int'):
  # Count max node cellid
  res = np.zeros(shape=nb_nodes, dtype=types.np_int_type)
  domain_compute.count_max_node_cellid(cells, res)
  max_node_cellid = np.max(res)

  # Create node cellid
  node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=types.np_int_type)
  domain_compute.create_node_cellid(cells, node_cellid)
  return node_cellid

# LocalDomainClass.py
def create_node_phyid(phy_faces: 'int[:, :]', nb_nodes: 'int'):
  # Count max node boundary faces
  # Create node boundary faceid
  return create_node_cellid(phy_faces, nb_nodes)

# LocalDomainClass.py
def create_cell_cellnid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
  # Count max cell cellnid
  i_visited = np.ones(cells.shape[0], dtype=types.np_int_type) * -1
  max_cell_cellnid = domain_compute.count_max_cell_cellnid(cells, node_cellid, i_visited)

  # Create cell cellnid
  cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=types.np_int_type)
  domain_compute.create_cell_cellnid(cells, node_cellid, cell_cellnid)
  return cell_cellnid

# Partitioning.py
def get_max_phyid(nb_cells: 'int', phy_faces: 'int[:, :]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]'):
  i_visited = np.ones(shape=nb_cells, dtype=types.np_int_type) * -1
  cell_nb_phyid = np.zeros(shape=nb_cells, dtype=types.np_int_type)

  domain_compute.get_cell_nb_phyid(phy_faces, node_cellid, i_visited, cell_nb_phyid)
  node_max_phyid = np.max(node_phyid[:, -1])
  cell_max_phyid = np.max(cell_nb_phyid)
  return node_max_phyid, cell_max_phyid

# Partitioning.py
def define_node_oldname(phy_faces, phy_faces_name, nb_nodes):
  node_oldname = np.zeros(shape=nb_nodes, dtype=types.np_int_type)
  domain_compute.define_node_oldname(phy_faces, phy_faces_name, node_oldname)

  return node_oldname

# Partitioning.py
def create_cellfid(
  cells: 'int[:, :]',
  node_cellid: 'int[:, :]',
  cell_type: 'int[:]',
  max_cell_faceid: 'int',
  max_face_nodeid: 'int'
):
  nb_cells = len(cells)
  # tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=types.np_int_type)
  # tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=types.np_int_type)
  cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=types.np_int_type)

  domain_compute.create_cellfid(
    cells,
    node_cellid,
    cell_type,
    max_cell_faceid,
    max_face_nodeid,
    cell_cellfid
  )

  return cell_cellfid