import numpy as np
import h5py

def _reinterpret_int32_as_float32(i: 'int32'):
  return np.int32(i).view(np.float32)

def _reinterpret_float32_as_int32(i: 'float32'):
  return np.float32(i).view(np.int32)

def _binary_search(array: 'int32[:]', item: 'int32') -> 'int32':
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

def _intersect_nodes(face_nodes: 'int32[:]', nb_nodes: 'int32', node_cellid: 'int32[:, :]',
                     intersect_cell: 'int32[:]'):
  """
    Get the common cells of neighboring cells of the face's nodes.

    Details:
    Identify the neighboring cells associated with each of the nodes that belong to a specific face.
    After identifying the neighboring cells for each of these nodes, we are interested in finding the common cells that are shared among all these neighboring cells.

    Args:
      face_nodes: nodes of the face
      nb_nodes : number of nodes of the face
      node_cellid: for each node get the neighbor cells

    Return:
      intersect_cell: array(2) common cells between all neighbors of each node (two at most)
  """
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


def _create_bf_cellid(phy_faces: 'int32[:, :]', node_cellid: 'int32[:, :]',
                      phyid_to_faceid: 'int32[:]', cell_faceid: 'int32[:, :]', intersect: 'int32[:]', bf_cellid: 'int32[:, :]'):
  #! Here a boundary cell is cell that is connected to a physical face.
  #! It is different from a boundary cell that has a node that has at least one neighbor physical face.
  counter = 0
  for phyid in range(len(phy_faces)):
    phy_face = phy_faces[phyid]
    size = phy_face[-1]
    _intersect_nodes(phy_face, size, node_cellid, intersect)
    cellid = intersect[0]
    faceid = phyid_to_faceid[phyid]
    if cellid == -1:
      raise RuntimeError("cellid must exist for a physical face")
    if faceid == -1:
      raise RuntimeError("faceid must exist for a physical face")
    face_index = -1
    for j in range(cell_faceid[cellid, -1]):
      if cell_faceid[cellid, j] == faceid:
        face_index = j
        break
    if face_index == -1:
      raise RuntimeError("faceid must exist in cell_faceid")
    bf_cellid[counter, 0] = cellid
    bf_cellid[counter, 1] = face_index
    counter += 1

def _create_ghost_info_3d(bf_cellid: 'int32[:, :]', cell_center: 'float[:, :]', cell_faceid: 'int32[:, :]', face_oldname: 'int32[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', ghost_info: 'float[:, :]'):
  # ghost_info [0=bc, 1=bf, 2=ghostcenter_x, 3=ghostcenter_y, 4=ghostcenter_z, 5=gamma, 6=face_oldname, 7=face_center_x, 8=face_center_y, 9=face_center_z, 10=face_normal_x, 11=face_normal_y, 12=face_normal_z]

  for i in range(bf_cellid.shape[0]):
    cid = bf_cellid[i, 0]
    bf = bf_cellid[i, 1]  # index of the face in the cell
    fid = cell_faceid[cid, bf]

    f_center = face_center[fid]
    c_center = cell_center[cid]
    f_normal = face_normal[fid]
    n_hat = f_normal / np.linalg.norm(f_normal)
    ghostcenter = c_center - 2 * np.dot(c_center - f_center, n_hat) * n_hat

    c_center = cell_center[cid]
    u = face_center[i] - c_center
    n = face_normal[fid] / face_measure[i]
    gamma = np.dot(u, n)

    ghost_info[i, 0] = _reinterpret_int32_as_float32(cid)
    ghost_info[i, 1] = _reinterpret_int32_as_float32(bf)
    ghost_info[i, 2] = ghostcenter[0]
    ghost_info[i, 3] = ghostcenter[1]
    ghost_info[i, 4] = ghostcenter[2]
    ghost_info[i, 5] = gamma
    ghost_info[i, 6] = _reinterpret_int32_as_float32(face_oldname[fid])
    ghost_info[i, 7] = face_center[fid, 0]  # fc_x
    ghost_info[i, 8] = face_center[fid, 1]  # fc_y
    ghost_info[i, 9] = face_center[fid, 2]  # fc_z
    ghost_info[i, 10] = face_normal[fid, 0]  # fn_x
    ghost_info[i, 11] = face_normal[fid, 1]  # fn_y
    ghost_info[i, 12] = face_normal[fid, 2]  # fn_z

class Locals:
  def __init__(self):
    self.map_cells = {}
    self.map_faces = {}
    self.map_nodes = {}
    self.nb_nodes = 0
    self.nb_cells = 0
    self.nb_faces = 0
    self.cells = np.zeros(shape=(1, 1), dtype=np.int32)
    self.faces = np.zeros(shape=(1, 1), dtype=np.int32)
    self.nodes = np.zeros(shape=(1, 1), dtype=np.float64)
    self.cell_faceid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cells_loctoglob = np.zeros(shape=1, dtype=np.int32)
    self.faces_loctoglob = np.zeros(shape=1, dtype=np.int32)
    self.nodes_loctoglob = np.zeros(shape=1, dtype=np.int32)
    self.face_cellid = np.ones(shape=1, dtype=np.int32)
    self.cell_cellnid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cell_halonid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cell_cellfid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cell_halofid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.node_cellid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.node_halonid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.node_name = np.zeros(shape=1, dtype=np.int32)
    self.halos_halosext = np.zeros(shape=1, dtype=np.int32)
    self.halos_halosint = {} # map(partition_id, int_cells)
    self.node_ghostnid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.node_haloghostnid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cell_ghostnid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.cell_haloghostnid = np.zeros(shape=(1, 1), dtype=np.int32)
    self.face_ghostid = np.zeros(shape=1, dtype=np.int32)

class HybridTestTables:
  def __init__(self, cell_loctoglob, dim):
    path = 'hybrid_test_tables.hd5'
    with h5py.File(path, 'r') as f:
      self.cells = f['cells'][...]
      self.cell_center = f['cell_center'][...]
      self.cell_volume = f['cell_volume'][...]
      self.cell_cellfid = f['cell_cellfid'][...]
      self.cell_cellnid = f['cell_cellnid'][...]
      self.cell_faceid = f['cell_faceid'][...]
      self.faces = f['faces'][...]
      self.face_cellid = f['face_cellid'][...]
      self.face_center = f['face_center'][...]
      self.face_oldname = f['face_oldname'][...]
      self.phyid_to_faceid = f['phyid_to_faceid'][...]
      self.phy_faces = f['phy_faces'][...]
      self.face_normal = f['face_normal'][...]
      self.face_measure = f['face_measure'][...]
      self.face_tangent = np.array([]) if dim == 2 else f['face_tangent'][...]
      self.face_binormal = np.array([]) if dim == 2 else f['face_binormal'][...]
      self.nodes = f['nodes'][...]
      self.node_cellid = f['node_cellid'][...]
      self.node_oldname = f['node_oldname'][...]
      #self.ghost_info = f['shared_ghost_info'][...] # indexed by physical id
      self.cell_ghostnid = f['cell_ghostnid'][...]
      self.node_ghostid = f['node_ghostid'][...]

    self.float_precision = 'float32' # TODO
    self.nb_cells = len(self.cells)
    self.nb_nodes = len(self.nodes)
    self.nb_faces = len(self.faces)
    self.part_vert = self._get_part_vert(cell_loctoglob)
    self.dim = dim
    self.nb_parts = max(self.part_vert) + 1
    self.face_ghostid = self._create_face_to_phyid(self.nb_faces, self.phyid_to_faceid)
    self.ghost_info = self._create_ghost_info(self.phy_faces, self.node_cellid, self.phyid_to_faceid, self.cell_center, self.cell_faceid, self.face_oldname, self.face_normal, self.face_center, self.face_measure)
    self.nb_phyid = len(self.ghost_info)
    self.locals = self._create_local(self.part_vert, self.cells, self.cell_faceid, self.faces, self.nodes, self.nb_parts, self.ghost_info, self.face_cellid, self.cell_cellnid, self.cell_cellfid, self.node_cellid, self.node_ghostid, self.node_oldname, self.cell_ghostnid, self.face_ghostid)


  @staticmethod
  def _create_face_to_phyid(nb_faces, phyid_to_faceid: 'int32[:]'):
    face_to_phyid = np.ones(shape=nb_faces, dtype=np.int32) * -1
    face_to_phyid[phyid_to_faceid] = np.arange(phyid_to_faceid.shape[0])
    return face_to_phyid

  def _get_part_vert(self, cell_loctoglob):
    part_vert = np.zeros(shape=self.nb_cells, dtype=np.int32)
    nb_partitions = len(cell_loctoglob)

    for p in range(nb_partitions):
      loctoglob = cell_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        part_vert[global_index] = p
    return part_vert

  def _create_ghost_info(self, phy_faces, node_cellid, phyid_to_faceid, cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]'):

    intersect = np.zeros(shape=2, dtype=np.int32)
    bf_cellid = np.zeros(shape=(len(phy_faces), 2), dtype=np.int32)
    _create_bf_cellid(phy_faces, node_cellid, phyid_to_faceid, cell_faceid, intersect, bf_cellid)


    ghost_info_data_size = 13
    ghost_info = np.zeros(shape=(len(phy_faces), ghost_info_data_size), dtype=self.float_precision)

    _create_ghost_info_3d(bf_cellid, cell_center, cell_faceid, face_oldname, face_normal, face_center, face_measure, ghost_info)

    return ghost_info

  @staticmethod
  def _create_local(part_vert, cells, cell_faceid, faces, nodes, nb_parts, ghost_info, face_cellid, cell_cellnid, cell_cellfid, node_cellid, node_ghostnid, node_oldname, cell_ghostnid, face_ghostid):
    l = [Locals() for _ in range(nb_parts)]

    for i in range(len(cells)):
      p = part_vert[i]
      l[p].map_cells[i] = len(l[p].map_cells)
      for j in range(cell_faceid[i, -1]):
        face_id = cell_faceid[i, j]
        if face_id not in l[p].map_faces:
          l[p].map_faces[face_id] = len(l[p].map_faces)
      for j in range(cells[i, -1]):
        node_id = cells[i, j]
        if node_id not in l[p].map_nodes:
          l[p].map_nodes[node_id] = len(l[p].map_nodes)

    for p in range(nb_parts):
      nb_cells = len(l[p].map_cells)
      nb_faces = len(l[p].map_faces)
      nb_nodes = len(l[p].map_nodes)
      l[p].nb_cells = nb_cells
      l[p].nb_faces = nb_faces
      l[p].nb_nodes = nb_nodes
      l[p].cells = np.zeros(shape=(nb_cells, cells.shape[1]), dtype=np.int32)
      l[p].cell_faceid = np.zeros(shape=(nb_cells, cell_faceid.shape[1]), dtype=np.int32)
      l[p].faces = np.zeros(shape=(nb_faces, faces.shape[1]), dtype=np.int32)
      l[p].nodes = np.zeros(shape=(nb_nodes, nodes.shape[1]), dtype=nodes.dtype)
      l[p].cells_loctoglob = np.zeros(shape=nb_cells, dtype=np.int32)
      l[p].faces_loctoglob = np.zeros(shape=nb_faces, dtype=np.int32)
      l[p].nodes_loctoglob = np.zeros(shape=nb_nodes, dtype=np.int32)

    def copy(dest, src, dic):
      size = src[-1]
      dest[0:size] = np.vectorize(dic.get)(src[0:size])
      dest[-1] = size

    def copy_lambda(dest, src, lambda_func):
      size = src[-1]
      if size != 0:
        values = np.vectorize(lambda_func)(src[0:size])
        values = values[values != -1]
        dest[0:len(values)] = values
        dest[-1] = len(values)
      else:
        dest[-1] = 0

    for p in range(nb_parts):
      for g_id, l_id in l[p].map_cells.items():
        copy(l[p].cells[l_id], cells[g_id], l[p].map_nodes)
        copy(l[p].cell_faceid[l_id], cell_faceid[g_id], l[p].map_faces)
        l[p].cells_loctoglob[l_id] = g_id
      for g_id, l_id in l[p].map_faces.items():
        copy(l[p].faces[l_id], faces[g_id], l[p].map_nodes)
        l[p].faces_loctoglob[l_id] = g_id
      for g_id, l_id in l[p].map_nodes.items():
        l[p].nodes[l_id, :] = nodes[g_id, :]
        l[p].nodes_loctoglob[l_id] = g_id

    for p in range(nb_parts):
      # face_cellid by global neighbor index
      l[p].face_cellid = np.ones(shape=(l[p].nb_faces, face_cellid.shape[1]), dtype=np.int32) * -1
      for i in range(l[p].nb_faces):
        g_id = l[p].faces_loctoglob[i]
        tmp = np.vectorize(lambda x: x if x != -1 and part_vert[x] == p else -1)(face_cellid[g_id, :])
        tmp = tmp[tmp != -1]
        l[p].face_cellid[i][0:len(tmp)] = tmp[:]

      # cell_cellnid
      # cell_cellfid
      # cell_halofid
      # cell_halonid
      l[p].cell_cellnid = np.zeros(shape=(l[p].nb_cells, cell_cellnid.shape[1]), dtype=np.int32)
      l[p].cell_halonid = np.zeros(shape=(l[p].nb_cells, cell_cellnid.shape[1]), dtype=np.int32)
      l[p].cell_cellfid = np.zeros(shape=(l[p].nb_cells, cell_cellfid.shape[1]), dtype=np.int32)
      l[p].cell_halofid = np.zeros(shape=(l[p].nb_cells, cell_cellfid.shape[1]), dtype=np.int32)
      for i in range(l[p].nb_cells):
        g_id = l[p].cells_loctoglob[i]
        copy_lambda(l[p].cell_cellnid[i], cell_cellnid[g_id], lambda x: x if part_vert[x] == p else -1)
        copy_lambda(l[p].cell_halonid[i], cell_cellnid[g_id], lambda x: x if part_vert[x] != p else -1)
        copy_lambda(l[p].cell_cellfid[i], cell_cellfid[g_id], lambda x: x if part_vert[x] == p else -1)
        copy_lambda(l[p].cell_halofid[i], cell_cellfid[g_id], lambda x: x if part_vert[x] != p else -1)
      # node_cellid
      # node_halonid
      l[p].node_cellid = np.zeros(shape=(l[p].nb_nodes, node_cellid.shape[1]), dtype=np.int32)
      l[p].node_halonid = np.zeros(shape=(l[p].nb_nodes, node_cellid.shape[1]), dtype=np.int32)
      l[p].node_name = np.zeros(shape=l[p].nb_nodes, dtype=np.int32)
      for i in range(l[p].nb_nodes):
        g_id = l[p].nodes_loctoglob[i]
        copy_lambda(l[p].node_cellid[i], node_cellid[g_id], lambda x: x if part_vert[x] == p else -1)
        copy_lambda(l[p].node_halonid[i], node_cellid[g_id], lambda x: x if part_vert[x] != p else -1)
        l[p].node_name[i] = node_oldname[g_id]
        if l[p].node_halonid[i, -1] != 0:
          l[p].node_name[i] = 10
      # halos_halosext
      # halos_halosint => map(partition_id, int_cells)
      # halos_neigh (implicit -> halos_haloint)
      # halos_centvol (implicit -> halos_halosext)
      # halos_sizehaloghost (implicit -> sum(node of haloghost))
      l[p].halos_halosext = set()
      l[p].halos_halosint = {}
      for i in range(l[p].nb_cells):
        g_id = l[p].cells_loctoglob[i]
        for j in range(l[p].cell_halonid[i, -1]):
          neighbor_cell = l[p].cell_halonid[i, j]
          neighbor_part = part_vert[neighbor_cell]
          # already neighbor_part != p
          l[p].halos_halosext.add(neighbor_cell)
          if neighbor_part not in l[p].halos_halosint:
            l[p].halos_halosint[neighbor_part] = set()
          l[p].halos_halosint[neighbor_part].add(g_id)
      l[p].halos_halosext = np.array(list(l[p].halos_halosext), dtype=np.int32)
      for key in l[p].halos_halosint:
        l[p].halos_halosint[key] = np.array(list(l[p].halos_halosint[key]), dtype=np.int32)

      #node_ghostnid
      #node_haloghostnid
      #cell_ghostnid
      #cell_haloghostnid
      #face_ghostid
      l[p].node_ghostnid = np.zeros(shape=(l[p].nb_nodes, node_ghostnid.shape[1]), dtype=np.int32)
      l[p].node_haloghostnid = np.zeros(shape=(l[p].nb_nodes, node_ghostnid.shape[1]), dtype=np.int32)
      l[p].cell_ghostnid = np.zeros(shape=(l[p].nb_cells, cell_ghostnid.shape[1]), dtype=np.int32)
      l[p].cell_haloghostnid = np.zeros(shape=(l[p].nb_cells, cell_ghostnid.shape[1]), dtype=np.int32)
      l[p].face_ghostid = np.zeros(shape=l[p].nb_faces, dtype=np.int32)
      for g_id, l_id in l[p].map_nodes.items():
        copy_lambda(l[p].node_ghostnid[l_id], node_ghostnid[g_id], lambda x: x if part_vert[_reinterpret_float32_as_int32(ghost_info[x, 0])] == p else -1)
        copy_lambda(l[p].node_haloghostnid[l_id], node_ghostnid[g_id], lambda x: x if part_vert[_reinterpret_float32_as_int32(ghost_info[x, 0])] != p else -1)
      for g_id, l_id in l[p].map_cells.items():
        copy_lambda(l[p].cell_ghostnid[l_id], cell_ghostnid[g_id], lambda x: x if part_vert[_reinterpret_float32_as_int32(ghost_info[x, 0])] == p else -1)
        copy_lambda(l[p].cell_haloghostnid[l_id], cell_ghostnid[g_id], lambda x: x if part_vert[_reinterpret_float32_as_int32(ghost_info[x, 0])] != p else -1)


      l[p].face_ghostid[:] = face_ghostid[np.array(list(l[p].map_faces.keys()))]

    return l
