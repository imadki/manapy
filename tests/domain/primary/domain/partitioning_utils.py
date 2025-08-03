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

    self.set_phyids = Dict.empty(key_type=int_type, value_type=types.boolean)
    self.map_bf_recv = Dict.empty(key_type=int_type, value_type=int_type)
    self.vec_shared_bf_recv = List.empty_list(int_type)
    self.vec_shared_bf_send = List.empty_list(int_type)
    self.set_halo_phyid_neighsub = Dict.empty(key_type=int_type, value_type=types.boolean)
    self.map_halo_neighsub = Dict.empty(key_type=int_type, value_type=types.ListType(int_type))
    self.map_halo_int = Dict.empty(key_type=int_type, value_type=types.ListType(int_type))



def create_sub_domains(part_vert, node_cellid, node_phyid, cells, cells_type, nodes, phy_faces, phy_faces_name, nb_parts, local_domains: 'LocalDomainStruct[:]'):
  i_visited = np.zeros(shape=len(cells), dtype=np.int32)
  vec_node_oldname = np.zeros(shape=len(nodes), dtype=np.int32)
  intersect_cell = np.zeros(shape=2, dtype=np.int32)
  boundary_cells = np.zeros(shape=len(phy_faces), dtype=np.int32) # cells that has at least one physical face attached to it
  part_phyid = np.zeros(shape=len(boundary_cells), dtype=np.int32)

  # #########################################################
  # Create Physical faces And node old name
  # #########################################################

  total_nb_phyfaces = 0
  for i in range(phy_faces.shape[0]):
    phy_face = phy_faces[i]
    name = phy_faces_name[i]
    size = phy_face[-1]
    _intersect_nodes(phy_face, size, node_cellid, intersect_cell)
    if intersect_cell[0] != -1:
      p = part_vert[intersect_cell[0]]
      local_domains[p].max_phy_face_nodeid = max(size, local_domains[p].max_phy_face_nodeid)
      local_domains[p].map_phy_faces[i] = len(local_domains[p].map_phy_faces)
      boundary_cells[total_nb_phyfaces] = intersect_cell[0] # cells may duplicate in boundary_cells
      total_nb_phyfaces += 1
    for j in range(size):
      nodeid = phy_face[j]
      if vec_node_oldname[nodeid] == 0 or vec_node_oldname[nodeid] > name:
        vec_node_oldname[nodeid] = name
  if total_nb_phyfaces != phy_faces.shape[0]:
    raise RuntimeError(f"Error: not all the physical faces match the domain faces ! {total_nb_phyfaces} {phy_faces.shape[0]}")

  # #########################################################
  # Create part_phyid
  # #########################################################
  for phyid in range(len(boundary_cells)):
    cell_id = boundary_cells[phyid]
    part_phyid[phyid] = part_vert[cell_id]

  # #########################################################
  # Create local cells and nodes, map_halos
  # #########################################################
  for i in range(cells.shape[0]):
    p = part_vert[i]

    cell_type = cells_type[i]
    max_info = get_max_info(cell_type)
    local_domains[p].max_cell_faceid = max(max_info[0], local_domains[p].max_cell_faceid)
    local_domains[p].max_face_nodeid = max(max_info[1], local_domains[p].max_face_nodeid)
    local_domains[p].max_cell_nodeid = max(max_info[2], local_domains[p].max_cell_nodeid)

    map_cells = local_domains[p].map_cells
    map_nodes = local_domains[p].map_nodes
    map_halos = local_domains[p].map_halos
    map_halo_int = local_domains[p].map_halo_int
    set_phyid = local_domains[p].set_phyids
    set_halo_phyid_neighsub = local_domains[p].set_halo_phyid_neighsub

    map_cells[i] = len(map_cells)
    for j in range(cells[-1]):
      nodeid = cells[i, j]


      # Determine max_cell_halonid, Create HaloCellMap, Create halo_interior_map, nb_node_halos
      nb_cell_halonid = 0
      for k in range(node_cellid[nodeid]):
        nb_node_halonid = 0
        n_cellid = node_cellid[k]
        if n_cellid != i and i_visited[n_cellid] != i:
          # allow visiting n_cellid only once for the current cell `i`
          i_visited[n_cellid] = i

          part_n_cellid = part_vert[n_cellid]
          if p != part_n_cellid:
            nb_cell_halonid += 1
            nb_node_halonid += 1
            if n_cellid not in map_halos:
              map_halos[n_cellid] = len(map_halos)

            # halo_interior
            if part_n_cellid not in map_halo_int:
              map_halo_int[part_n_cellid] = List.empty_list(int_type)
            vec = map_halo_int[part_n_cellid]
            if len(vec) == 0 or vec[-1] != i:
              map_halo_int.append(i) # append haloint_cell `i` to halo interiors connected to neighbor part `part_n_cellid`

        if nodeid not in map_nodes:
          local_domains[p].nb_node_halos += nb_node_halonid
          if nb_node_halonid != 0:
            local_domains[p].nb_node_halos += 2 # 1 for node and 1 for counter ==> 2
          map_nodes[nodeid] = len(map_nodes)
      local_domains[p].max_cell_halonid = max(nb_cell_halonid, local_domains[p].max_cell_halonid)

      # max_node_halophyid, set_phyid, set_halo_phyid_neighsub
      nb_node_halophyid = 0
      for k in range(node_phyid[nodeid, -1]):
        phy_id = node_phyid[nodeid, k]
        phy_id_part = part_phyid[phy_id]
        if phy_id not in set_phyid:
          set_phyid[phy_id] = True # collect all phyid that is related to partition `p`
        if p != phy_id_part:
          nb_node_halophyid += 1
          set_halo_phyid_neighsub[phy_id_part] = True
      local_domains[p].max_node_halophyid = max(nb_node_halophyid, local_domains[p].max_node_halophyid)








