import numpy as np
import numba
from numba.typed import Dict, List
from numba import types

int_type = types.int32
CELL_TRIANGLE = 0
CELL_QUAD = 1
CELL_TETRA = 2
CELL_HEXA = 3
CELL_PYRAMID = 4

def _binary_search(array: 'int[:]', item: 'int') -> 'int':
  """
    Check if an item is in the array
    Return index >= 0 if the item is in the array otherwise -1

    Note:
      The number of item in the array must be array[-1]
  """
  size = array[-1]
  left = 0
  right = size - 1

  while left <= right:
    mid = (left + right) // 2
    mid_val = array[mid]

    if mid_val == item:
      return mid
    elif mid_val < item:
      left = mid + 1
    else:
      right = mid - 1

  return -1

def _intersect_nodes(face_nodes: 'int[:]', nb_nodes: 'int', node_cellid: 'int[:, :]',
                     intersect_cell: 'int[:]'):
  index = 0

  intersect_cell[0] = -1
  intersect_cell[1] = -1

  cells = node_cellid[face_nodes[0]]
  for i in range(cells[-1]):
    intersect_cell[index] = cells[i]
    for j in range(1, nb_nodes):
      if _binary_search(node_cellid[face_nodes[j]], cells[i]) == -1:
        intersect_cell[index] = -1
        break
    if intersect_cell[index] != -1:
      index = index + 1
    if index >= 2:
      return

def get_max_info(cell_type):
  if cell_type == CELL_TRIANGLE:
    return [3, 2, 3]
  elif cell_type == CELL_QUAD:
    return [4, 2, 4]
  elif cell_type == CELL_TETRA:
    return [4, 3, 4]
  elif cell_type == CELL_HEXA:
    return [6, 4, 8]
  elif cell_type == CELL_PYRAMID:
    return [5, 4, 5]
  return [0, 0, 0]

class LocalDomainStruct:
  def __init__(self):
    self.nodes = None
    self.cells = None
    self.cell_type = None
    self.phy_faces = None
    self.phy_faces_name = None
    self.bf_cellid = None
    self.cell_loctoglob = None
    self.node_loctoglob = None
    self.node_oldname = None
    self.halo_neighsub = None
    self.node_halos = None
    self.node_halobfid = None
    self.shared_bf_recv = None
    self.bf_recv_part_size = None
    self.shared_bf_send = None
    self.halo_halosext = None
    self.halo_halosint = None
    self.max_cell_nodeid = 0
    self.max_cell_faceid = 0
    self.max_face_nodeid = 0
    self.max_node_haloid = 0
    self.max_cell_halonid = 0

    # Temporarily
    self.max_node_halophyid = 0
    self.max_phy_face_nodeid = 0
    self.nb_node_halos = 0
    self.map_cells = Dict.empty(key_type=int_type, value_type=int_type)
    self.map_phy_faces = Dict.empty(key_type=int_type, value_type=int_type)
    self.map_nodes = Dict.empty(key_type=int_type, value_type=int_type)
    self.map_halos = Dict.empty(key_type=int_type, value_type=int_type)
    self.set_bf_recv = Dict.empty(key_type=int_type, value_type=types.boolean)
    self.map_bf_recv = Dict.empty(key_type=int_type, value_type=int_type)
    self.vec_shared_bf_recv = List.empty_list(int_type)
    self.vec_shared_bf_send = List.empty_list(int_type)
    self.set_halo_bf_neighsub = Dict.empty(key_type=int_type, value_type=types.boolean)
    self.map_halo_neighsub = Dict.empty(key_type=int_type, value_type=types.ListType(int_type))
    self.map_halo_int = Dict.empty(key_type=int_type, value_type=types.ListType(int_type))



def create_sub_domains(part_vert, node_cellid, node_bfid, bf_cellid, cells, cell_cellnid, cells_type, nodes, phy_faces, phy_faces_name, nb_parts, local_domains: 'LocalDomainStruct[:]'):
  #
  #

  b_visited = np.zeros(shape=len(cells), dtype=np.int8)
  for i in range(cells.shape[0]):
    p = part_vert[i]

    cell_type = cells_type[i]
    max_info = get_max_info(cell_type)
    local_domains[p].max_cell_faceid = max(max_info[0], local_domains[p].max_cell_faceid)
    local_domains[p].max_face_nodeid = max(max_info[1], local_domains[p].max_face_nodeid)
    local_domains[p].max_cell_nodeid = max(max_info[2], local_domains[p].max_cell_nodeid)

    map_cells = local_domains[p].map_cells
    map_phy_faces = local_domains[p].map_phy_faces
    map_nodes = local_domains[p].map_nodes
    map_halos = local_domains[p].map_halos
    map_halo_neighsub = local_domains[p].map_halo_neighsub
    map_halo_int = local_domains[p].map_halo_int

    map_cells[i] = len(map_cells)
    for j in range(cells[-1]):
      nodeid = cells[i, j]
      if nodeid not in map_nodes:
        map_nodes[nodeid] = len(map_nodes)

      # Determine max_cell_halonid, Create HaloMap, Create HaloNeighDomain
      counternid = 0
      for k in range(node_cellid[nodeid]):
        n_cellid = node_cellid[k]
        if not b_visited[n_cellid]:
          b_visited[n_cellid] = True
          part_n_cellid = part_vert[n_cellid]
          if p != part_n_cellid:
            counternid += 1
            if n_cellid not in map_halos:
              map_halos[n_cellid] = len(map_halos)

            if part_n_cellid not in map_halo_neighsub:
              map_halo_neighsub[part_n_cellid] = [0, -1] # size, tmp
            if map_halo_neighsub[part_n_cellid][1] != i:
              map_halo_neighsub[part_n_cellid][0] += 1
              map_halo_neighsub[part_n_cellid][1] = i

            # halo_interior
            if len(map_halo_int) == 0 or map_halo_int[-1] != i:
              map_halo_int.append(i)

      local_domains[p].max_cell_halonid = max(counternid, local_domains[p].max_cell_halonid)


