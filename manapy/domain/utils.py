import numpy as np
from manapy.domain import compute

def create_node_cellid(cells: 'int[:, :]', nb_nodes: 'int'):
  # Count max node cellid
  res = np.zeros(shape=nb_nodes, dtype=np.int32)
  compute.count_max_node_cellid(cells, res)
  max_node_cellid = np.max(res)

  # Create node cellid
  node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32)
  compute.create_node_cellid(cells, node_cellid)
  return node_cellid

def create_node_phyid(phy_faces: 'int[:, :]', nb_nodes: 'int'):
  # Count max node boundary faces
  # Create node boundary faceid
  return create_node_cellid(phy_faces, nb_nodes)


def create_cell_cellnid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
  # Count max cell cellnid
  i_visited = np.ones(cells.shape[0], dtype=np.int32) * -1
  max_cell_cellnid = compute.count_max_cell_cellnid(cells, node_cellid, i_visited)

  # Create cell cellnid
  cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=np.int32)
  compute.create_cell_cellnid(cells, node_cellid, cell_cellnid)
  return cell_cellnid