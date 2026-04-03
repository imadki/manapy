import numpy as np
import h5py

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

class TestTables:
  def __init__(self, test_tables_path, part_vert, dim):
    path = test_tables_path
    with h5py.File(path, 'r') as f:
      self.cells = f['cells'][...]
      self.cell_center = f['cell_center'][...]
      self.cell_area = f['cell_volume'][...]
      self.cell_cellfid = f['cell_cellfid'][...]
      self.cell_cellnid = f['cell_cellnid'][...]
      self.cell_faceid = f['cell_faceid'][...]
      self.faces = f['faces'][...]
      self.face_cellid = f['face_cellid'][...]
      self.face_center = f['face_center'][...]
      self.face_oldname = f['face_oldname'][...]
      self.phy_faces = f['phy_faces'][...]
      self.face_normal = f['face_normal'][...]
      self.face_measure = f['face_measure'][...]
      self.face_tangent = np.array([]) if dim == 2 else f['face_tangent'][...]
      self.face_binormal = np.array([]) if dim == 2 else f['face_binormal'][...]
      self.nodes = f['nodes'][...]
      self.node_cellid = f['node_cellid'][...]
      self.node_oldname = f['node_oldname'][...]
      self.ghost_info_flt = f['shared_ghost_info_flt'][...] # indexed by physical id
      self.ghost_info_int = f['shared_ghost_info_int'][...] # indexed by physical id
      self.cell_ghostnid = f['cell_ghostnid'][...] # point to self.ghost_info
      self.node_ghostid = f['node_ghostid'][...] # point to self.ghost_info
      self.face_ghostid = f['face_to_phyid'][...]  # point to self.ghost_info which is index physical id


    self.nb_cells = len(self.cells)
    self.nb_nodes = len(self.nodes)
    self.nb_faces = len(self.faces)
    self.part_vert = part_vert
    self.dim = dim
    self.nb_parts = max(self.part_vert) + 1
    self.nb_phyid = len(self.ghost_info_int)
    self.face_name = self._define_face_name(self.faces, self.face_cellid, self.part_vert, self.face_oldname)
    self.locals = self._create_local(self.part_vert, self.cells, self.cell_faceid, self.faces, self.nodes, self.nb_parts, self.ghost_info_int, self.face_cellid, self.cell_cellnid, self.cell_cellfid, self.node_cellid, self.node_ghostid, self.node_oldname, self.cell_ghostnid, self.face_ghostid)


  @staticmethod
  def _define_face_name(
          faces: 'int32[:, :]',
          face_cellid: 'int32[:]',
          part_vert: 'int32[:]',
          face_oldname: 'int32[:]'
  ):
    face_name = np.zeros(shape=faces.shape[0], dtype=np.int32)
    for i in range(faces.shape[0]):
      face_name[i] = face_oldname[i]
      if face_cellid[i, 1] != -1 and part_vert[face_cellid[i, 0]] != part_vert[face_cellid[i, 1]]:
        face_name[i] = 10
    return face_name


  @staticmethod
  def _create_local(part_vert, cells, cell_faceid, faces, nodes, nb_parts, ghost_info_int, face_cellid, cell_cellnid, cell_cellfid, node_cellid, node_ghostnid, node_oldname, cell_ghostnid, face_ghostid):
    l = [Locals() for _ in range(nb_parts)]

    # Determine Cells, Faces and nodes For evert partition `p`.
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

    # Create cells cell_faceid, faces, nodes, cells_loctolob, faces_loctoglob, nodes_loctoglob tables
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

    # Copy dest[0:src[-1]] = dic[src[0:src[-1]]
    def copy(dest, src, dic):
      size = src[-1]
      dest[0:size] = np.vectorize(dic.get)(src[0:size])
      dest[-1] = size

    # Copy with condition
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
        copy_lambda(l[p].node_ghostnid[l_id], node_ghostnid[g_id], lambda x: x if part_vert[ghost_info_int[x, 0]] == p else -1)
        copy_lambda(l[p].node_haloghostnid[l_id], node_ghostnid[g_id], lambda x: x if part_vert[ghost_info_int[x, 0]] != p else -1)
      for g_id, l_id in l[p].map_cells.items():
        copy_lambda(l[p].cell_ghostnid[l_id], cell_ghostnid[g_id], lambda x: x if part_vert[ghost_info_int[x, 0]] == p else -1)
        copy_lambda(l[p].cell_haloghostnid[l_id], cell_ghostnid[g_id], lambda x: x if part_vert[ghost_info_int[x, 0]] != p else -1)


      l[p].face_ghostid[:] = face_ghostid[np.array(list(l[p].map_faces.keys()))]

    return l
