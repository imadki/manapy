import numpy as np
import numba
from numba.typed import Dict, List
from numba import types
from LocalDomainStructData import new_local_domains, LocalDomainStructDataType, ListLocalDomainStructDataType
from compile_fun import compile


int_type = types.int32
CELL_TRIANGLE = 1
CELL_QUAD = 2
CELL_TETRA = 3
CELL_HEXA = 4
CELL_PYRAMID = 5

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

def _get_max_info(cell_type: 'int'):
  # [max_cell_faceid, max_face_nodeid, max_cell_nodeid]
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



def _create_sub_domains(part_vert: 'int[:]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]', cells: 'int[:, :]', cells_type: 'int[:]', phy_faces: 'int[:, :]', phy_faces_name: 'int[:]', local_domains: ListLocalDomainStructDataType, i_visited: 'int32[:]', vec_node_oldname: 'int[:]', intersect_cell: 'int[:]', boundary_cells: 'int[:]', part_phyid: 'int[:]'):

  # #########################################################
  # Create Physical faces And node old name, boundary_cells
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
  # map_cells, map_nodes, map_halos, map_halo_int, set_phyid, set_halo_phyid_neighsub
  # max_cell_faceid, max_face_nodeid, max_cell_nodeid, max_cell_halonid, nb_node_halos, max_node_halophyid
  # #########################################################
  for i in range(cells.shape[0]):
    p = part_vert[i]

    cell_type = cells_type[i]
    max_info = _get_max_info(cell_type)
    local_domains[p].max_cell_faceid = max(max_info[0], local_domains[p].max_cell_faceid)
    local_domains[p].max_face_nodeid = max(max_info[1], local_domains[p].max_face_nodeid)
    local_domains[p].max_cell_nodeid = max(max_info[2], local_domains[p].max_cell_nodeid)

    map_cells = local_domains[p].map_cells
    map_nodes = local_domains[p].map_nodes
    map_halos = local_domains[p].map_halos
    map_halo_int = local_domains[p].map_halo_int
    set_phyid = local_domains[p].set_phyids
    set_halo_phyid_neighsub = local_domains[p].set_halo_phyid_neighsub

    # local cells
    map_cells[i] = len(map_cells)

    # Determine max_cell_halonid, Create HaloCellMap, Create halo_interior_map, nb_node_halos
    nb_cell_halonid = 0
    for j in range(cells[i, -1]):
      nodeid = cells[i, j]
      nb_node_halonid = 0
      for k in range(node_cellid[nodeid, -1]):
        n_cellid = node_cellid[nodeid, k]
        part_n_cellid = part_vert[n_cellid]

        if p != part_n_cellid:
          nb_node_halonid += 1

          # halos
          if n_cellid not in map_halos:
            map_halos[n_cellid] = len(map_halos)

          # halo_interior
          if part_n_cellid not in map_halo_int:
            map_halo_int[part_n_cellid] = List.empty_list(int_type)
          vec = map_halo_int[part_n_cellid]
          if len(vec) == 0 or vec[-1] != i:
            vec.append(i) # append haloint_cell `i` to halo interiors connected to neighbor part `part_n_cellid`

          #nb_cell_halonid
          if n_cellid != i and i_visited[n_cellid] != i:
            # allow visiting n_cellid only once for the current cell `i`
            i_visited[n_cellid] = i
            nb_cell_halonid += 1


      if nodeid not in map_nodes:
        local_domains[p].nb_node_halos += nb_node_halonid
        if nb_node_halonid != 0:
          local_domains[p].nb_node_halos += 2 # 1 for node and 1 for counter ==> 2
        map_nodes[nodeid] = len(map_nodes)


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

    local_domains[p].max_cell_halonid = max(nb_cell_halonid, local_domains[p].max_cell_halonid)


def _create_tables(ld: LocalDomainStructDataType, nodes: 'float64[:, :]', part_phyid: 'int[:]'):
  # #########################################################
  # vec_phyids, map_phyids, nb_halos_int
  # #########################################################
  set_phyids = ld.set_phyids
  vec_phyids = np.zeros(shape=len(set_phyids), dtype=np.int32)
  map_phyids = Dict.empty(key_type=int_type, value_type=int_type)
  nb_halos_int = 0

  counter = 0
  for key in set_phyids:
    vec_phyids[counter] = key
    counter += 1
  vec_phyids = vec_phyids[np.argsort(part_phyid[vec_phyids])]
  for i in range(len(vec_phyids)):
    item = vec_phyids[i]
    map_phyids[item] = i

  for item in ld.map_halo_int:
    nb_halos_int += len(ld.map_halo_int[item])
  # #########################################################
  # Tables
  # #########################################################
  ld.map_phyids = map_phyids
  ld.vec_phyids = vec_phyids
  ld.nodes = np.zeros(shape=(len(ld.map_nodes), nodes.shape[1]), dtype=np.float64)
  ld.cells = np.zeros(shape=(len(ld.map_cells), ld.max_cell_nodeid + 1), dtype=np.int32)
  ld.cells_type = np.zeros(shape=len(ld.map_cells), dtype=np.int8)
  ld.phy_faces = np.zeros(shape=(len(ld.map_phy_faces), ld.max_phy_face_nodeid + 1), dtype=np.int32)
  ld.phy_faces_name = np.zeros(shape=len(ld.map_phy_faces), dtype=np.int32)
  ld.cell_loctoglob = np.zeros(shape=len(ld.map_cells), dtype=np.int32)
  ld.node_loctoglob = np.zeros(shape=len(ld.map_nodes), dtype=np.int32)
  ld.node_oldname = np.zeros(shape=len(ld.map_nodes), dtype=np.int32)
  ld.halo_neighsub = np.zeros(shape=(2, len(ld.map_halo_int)), dtype=np.int32)
  ld.node_halos = np.zeros(shape=ld.nb_node_halos, dtype=np.int32)
  ld.node_halophyid = np.zeros(shape=(len(ld.map_nodes), ld.max_node_halophyid + 1), dtype=np.int32)
  ld.halo_halosext = np.zeros(shape=(len(ld.map_halos), ld.max_cell_nodeid + 2), dtype=np.int32)
  ld.halo_halosint = np.zeros(shape=nb_halos_int, dtype=np.int32)
  ld.phyid_recv = np.zeros(shape=len(ld.vec_phyids), dtype=np.int32)
  ld.phyid_recv_part_size = np.zeros(shape=(len(ld.set_halo_phyid_neighsub) * 2 + 2), dtype=np.int32)

def _create_locals(p: 'int', cells: 'int[:, :]', nodes: 'float64[:, :]', cells_type: 'int[:]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]', part_phyid: 'int[:]', phy_faces: 'int[:, :]', phy_faces_name: 'int[:]', part_vert: 'int[:]', vec_node_oldname: 'int[:]', local_domain: LocalDomainStructDataType):
  map_cells = local_domain.map_cells
  map_nodes = local_domain.map_nodes
  map_halos = local_domain.map_halos
  map_phy_faces = local_domain.map_phy_faces
  map_halo_int = local_domain.map_halo_int
  l_cells = local_domain.cells
  l_cell_loctoglob = local_domain.cell_loctoglob
  l_cells_type = local_domain.cells_type
  l_nodes = local_domain.nodes
  l_node_loctoglob = local_domain.node_loctoglob
  l_node_oldname = local_domain.node_oldname
  l_node_halos = local_domain.node_halos
  l_node_halophyid = local_domain.node_halophyid
  l_phy_faces = local_domain.phy_faces
  l_phy_faces_name = local_domain.phy_faces_name
  l_halo_neighsub = local_domain.halo_neighsub
  l_halo_halosint = local_domain.halo_halosint
  l_halo_halosext = local_domain.halo_halosext

  map_phyids = local_domain.map_phyids
  vec_phyids = local_domain.vec_phyids
  phyid_recv = local_domain.phyid_recv
  phyid_recv_part_size = local_domain.phyid_recv_part_size



  # #########################################################
  # l_cells, l_cells_type, l_cell_loctoglob
  # #########################################################

  for g_id in map_cells:
    l_id = map_cells[g_id]

    l_cells_type[l_id] = cells_type[g_id] # cell_type
    l_cell_loctoglob[l_id] = g_id # cell_loctoglob

    for j in range(cells[g_id, -1]):
      nodeid = cells[g_id, j]
      l_cells[l_id, j] = map_nodes[nodeid] # get local nodeid
    l_cells[l_id, -1] = cells[g_id, -1] # size

  # #########################################################
  # l_nodes, l_node_loctoglob, l_node_oldname, l_node_halos, l_node_halophyid, max_node_haloid
  # #########################################################
  halos_counter = 0
  for g_id in map_nodes:
    l_id = map_nodes[g_id]

    l_node_loctoglob[l_id] = g_id # node_loctoglob
    l_node_oldname[l_id] = vec_node_oldname[g_id] # node_oldname

    for j in range(nodes.shape[1]):
      l_nodes[l_id, j] = nodes[g_id, j] # copy coordinate (x, y, z)

    node_counter = -1
    for j in range(node_cellid[g_id, -1]):
      neighbor_cell = node_cellid[g_id, j]
      neighbor_part = part_vert[neighbor_cell]
      if p != neighbor_part:
        if node_counter == -1:
          l_node_halos[halos_counter] = l_id # node_id
          l_node_halos[halos_counter + 1] = 0 # size
          node_counter = halos_counter + 1 # pointer to size
          halos_counter += 2
        l_node_halos[halos_counter] = map_halos[neighbor_cell]
        halos_counter += 1
        l_node_halos[node_counter] += 1
    if node_counter != -1:
      local_domain.max_node_haloid = max(l_node_halos[node_counter], local_domain.max_node_haloid)

    # l_node_halophyid
    counter = 0
    for j in range(node_phyid[g_id, -1]):
      neighbor_phyid = node_phyid[g_id, j]
      neighbor_part = part_phyid[neighbor_phyid]
      if p != neighbor_part:
        l_node_halophyid[l_id, counter] = map_phyids[neighbor_phyid]
        counter += 1
    l_node_halophyid[l_id, -1] = counter


  # #########################################################
  # l_phy_faces, l_phy_faces_name
  # #########################################################

  for g_id in map_phy_faces:
    l_id = map_phy_faces[g_id]

    l_phy_faces_name[l_id] = phy_faces_name[g_id] # phy_faces_name

    for j in range(phy_faces[g_id, -1]):
      nodeid = phy_faces[g_id, j]
      l_phy_faces[l_id, j] = map_nodes[nodeid] # local nodeid
    l_phy_faces[l_id, -1] = phy_faces[g_id, -1] # size

  # #########################################################
  # l_halo_neighsub, l_halo_halosint, l_halo_halosext
  # #########################################################

  neighsub_counter = 0
  halosint_counter = 0
  for partition in map_halo_int:
    vect = map_halo_int[partition]
    l_halo_neighsub[0, neighsub_counter] = partition
    l_halo_neighsub[1, neighsub_counter] = len(vect)
    neighsub_counter += 1
    for interior_cell in vect:
      l_halo_halosint[halosint_counter] = interior_cell
      halosint_counter += 1

  for g_id in map_halos:
    l_id = map_halos[g_id]

    l_halo_halosext[l_id, 0] = g_id
    for j in range(cells[g_id, -1]):
      nodeid = cells[g_id, j]
      l_halo_halosext[l_id, j + 1] = nodeid
    l_halo_halosext[l_id, -1] = cells[g_id, -1] + 1

  # #########################################################
  # phyid_recv => [phyid of p_a, ..., phyid of p_b, ...]
  # phyid_recv_part_size => [partition Id, size, ...]
  # #########################################################

  old_part = -1
  counter = 0
  p_has_halo_phyid = False
  for i in range(len(vec_phyids)):
    g_id = vec_phyids[i]
    part = part_phyid[g_id]
    phyid_recv[i] = g_id
    if p == part:
      p_has_halo_phyid = True
      phyid_recv[i] = map_phy_faces[g_id] # transform phyid to local for p == part
    if old_part != part:
      phyid_recv_part_size[counter] = part
      phyid_recv_part_size[counter + 1] = 0
      old_part = part
      counter += 2
    phyid_recv_part_size[counter - 1] += 1
  if not p_has_halo_phyid:
    phyid_recv_part_size[counter] = p
    phyid_recv_part_size[counter + 1] = 0

def _create_phyid_send(local_domains: ListLocalDomainStructDataType):
  # #########################################################
  # phyid_send => [partition_id, size, indices point to phyid_recv_part_size, ...]
  # #########################################################
  for p in range(len(local_domains)):
    vec_phyids = local_domains[p].vec_phyids
    phyid_recv_part_size = local_domains[p].phyid_recv_part_size
    counter = 0

    for i in range(0, len(phyid_recv_part_size), 2):
      part = phyid_recv_part_size[i]
      size = phyid_recv_part_size[i + 1]
      if part != p:
        list_phyid_send = local_domains[part].list_phyid_send
        map_phyids = local_domains[part].map_phyids
        list_phyid_send.append(p)
        list_phyid_send.append(size)
        for j in range(size):
          phy_id = vec_phyids[counter + j]
          index = map_phyids[phy_id]
          list_phyid_send.append(index)
      counter += size

  for p in range(len(local_domains)):
    list_phyid_send = local_domains[p].list_phyid_send
    local_domains[p].phyid_send = np.zeros(shape=len(list_phyid_send), dtype=np.int32)
    for i in range(len(list_phyid_send)):
      local_domains[p].phyid_send[i] = list_phyid_send[i]

def _create_partition_tables(local_domains: ListLocalDomainStructDataType, cells: 'int[:, :]', nodes: 'float64[:, :]', cells_type: 'int[:]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]', part_phyid: 'int[:]', phy_faces: 'int[:, :]', phy_faces_name: 'int[:]', part_vert: 'int[:]', vec_node_oldname: 'int[:]', float_precision: 'int', dim: 'int'):
  for p in range(len(local_domains)):
    local_domain = local_domains[p]
    _create_tables(local_domain, nodes, part_phyid)
    _create_locals(p, cells, nodes, cells_type, node_cellid, node_phyid, part_phyid, phy_faces, phy_faces_name, part_vert, vec_node_oldname, local_domain)
    local_domains[p].dim = dim
    local_domains[p].float_precision = float_precision
  _create_phyid_send(local_domains)

def _create_local_domains(part_vert: 'int[:]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]', cells: 'int[:, :]', cells_type: 'int[:]', nodes: 'float64[:, :]', phy_faces: 'int[:, :]', phy_faces_name: 'int[:]', nb_parts: 'int', float_precision: 'int', dim: 'int'):

  local_domains = new_local_domains(nb_parts)
  i_visited = np.ones(shape=len(cells), dtype=np.int32) * -1
  vec_node_oldname = np.zeros(shape=len(nodes), dtype=np.int32)
  intersect_cell = np.zeros(shape=2, dtype=np.int32)
  boundary_cells = np.zeros(shape=len(phy_faces), dtype=np.int32)  # cells that has at least one physical face attached to it
  part_phyid = np.zeros(shape=len(boundary_cells), dtype=np.int32)

  _create_sub_domains(part_vert, node_cellid, node_phyid, cells, cells_type, phy_faces, phy_faces_name, local_domains, i_visited, vec_node_oldname, intersect_cell, boundary_cells, part_phyid)

  _create_partition_tables(local_domains, cells, nodes, cells_type, node_cellid, node_phyid, part_phyid, phy_faces, phy_faces_name, part_vert, vec_node_oldname, float_precision, dim)

  return local_domains

# def compile(func):
#   #return func
#   return numba.jit(nopython=True, fastmath=True, cache=True)(func)

# private
_binary_search = compile(_binary_search)
_intersect_nodes = compile(_intersect_nodes)
_get_max_info = compile(_get_max_info)
_create_tables = compile(_create_tables)
_create_locals = compile(_create_locals)
_create_phyid_send = compile(_create_phyid_send)
_create_sub_domains = compile(_create_sub_domains)
_create_partition_tables = compile(_create_partition_tables)

# public
create_local_domains = compile(_create_local_domains)