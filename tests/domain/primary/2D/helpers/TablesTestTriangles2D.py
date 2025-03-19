import numpy as np

class TablesTestTriangles2D:
  def __init__(self, float_precision, d_cell_loctoglob, g_cell_nodeid):
    """
    d_cell_loctoglob: loctoglob of the local domain
    g_cell_nodeid: cell_nodeid of the global domain
    """

    if float_precision == 'float32':
      float_precision = np.float32
    elif float_precision == 'float64':
      float_precision = np.float64
    else:
      raise ValueError('float_precision must be "float32" or "float64"')

    width = 10

    self.width = width # number of rectangles along the x-axis and y-axis
    self.WIDTH = 10.0
    self.HEIGHT = 5.0
    self.nb_faces = 320
    self.nb_cells = self.width * self.width * 2
    self.nb_nodes = 121
    self.nb_ghosts = 40

    self.x_length = self.WIDTH / self.width #of the square
    self.y_length = self.HEIGHT / self.width #of the square
    self.nb_partitions = len(d_cell_loctoglob)

    self.d_cell_loctoglob = d_cell_loctoglob
    self.g_cell_nodeid = g_cell_nodeid

    self.cell_vertices = np.zeros(shape=(self.nb_cells, 3, 2), dtype=float_precision)
    self.cell_center = np.zeros(shape=(self.nb_cells, 2), dtype=float_precision)
    self.cell_area = np.zeros(shape=(self.nb_cells), dtype=float_precision)
    self.cell_which_partition = np.ones(shape=(self.nb_cells), dtype=np.int32) * -1
    self.cell_halonid = np.ones(shape=(self.nb_cells, 17), dtype=np.int32) * -1
    self.cell_halofid = np.ones(shape=(self.nb_cells, 3), dtype=np.int32) * -1
    self.g_cell_cellnid = np.ones(shape=(self.nb_cells, 18), dtype=np.int32) * -1
    self.l_cell_cellnid = np.ones(shape=(self.nb_cells, 18), dtype=np.int32) * -1
    self.g_cell_cellfid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.l_cell_cellfid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.cell_nf = np.zeros(shape=(self.nb_cells, 4, 2), dtype=float_precision)

    self.faces_measure = np.zeros(shape=(self.nb_cells, 3), dtype=float_precision)
    self.face_center = np.zeros(shape=(self.nb_cells, 3, 2), dtype=float_precision)
    self.face_normal = np.zeros(shape=(self.nb_cells, 3, 2), dtype=float_precision)
    self.faces_vertices = np.zeros(shape=(self.nb_cells, 3, 2, 2), dtype=float_precision)
    self.face_ghostcenter = np.zeros(shape=(self.nb_cells, 3, 3), dtype=float_precision)
    self.g_face_name = np.ones(shape=(self.nb_cells, 3), dtype=np.int32) * -1
    self.l_face_name = np.ones(shape=(self.nb_cells, 3), dtype=np.int32) * -1
    self.g_face_cellid = np.ones(shape=(self.nb_cells, 3, 2), dtype=np.int32) * -1
    self.l_face_cellid = np.ones(shape=(self.nb_cells, 3, 2), dtype=np.int32) * -1
    self.face_nodeid = np.ones(shape=(self.nb_cells, 3, 2), dtype=np.int32) * -1

    self.g_node_cellid = np.ones(shape=(self.nb_nodes, 7), dtype=np.int32) * -1
    self.l_node_cellid = np.ones(shape=(self.nb_cells, 3, 7), dtype=np.int32) * -1
    self.node_halonid = np.ones(shape=(self.nb_cells, 3, 7), dtype=np.int32) * -1
    self.g_node_name = np.ones(shape=(self.nb_cells, 3), dtype=np.int32) * -1
    self.l_node_name = np.ones(shape=self.nb_nodes, dtype=np.int32) * -1

    self.ghost_info = np.ones(shape=(self.nb_ghosts, 6), dtype=float_precision) * -1
    self.g_cell_ghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    self.l_cell_ghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    self.cell_haloghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    self.face_ghostid = np.ones(shape=(self.nb_cells, 3), dtype=np.int32) * -1
    self.g_node_ghostid = np.ones(shape=(self.nb_nodes, 3), dtype=np.int32) * -1
    self.l_node_ghostid = np.ones(shape=(self.nb_cells, 3, 3), dtype=np.int32) * -1
    self.node_haloghostid = np.ones(shape=(self.nb_cells, 3, 3), dtype=np.int32) * -1

    self.halo_halosint = np.array([], np.int32)
    self.halo_neigh = np.zeros(shape=(self.nb_partitions, self.nb_partitions), dtype=np.int32)
    self.halo_sizehaloghost = np.zeros(shape=self.nb_partitions, dtype=np.int32)

    """
      cells.faceid
      Tested by comparing it with the reuslt of _set_face_vertices

      cells.halonid # used to get cell halo cells by node
      faces.halofid # used to get cell halo cells by face
      halos.halosext # used to get cell halo cells
      Tested by comparing with the results of _set_cell_halonid and _set_cell_halofid.

      cell.haloghostcenter
      Tested using cell.haloghostnid
    """

  ###################
  ## Face Info
  ###################
  """
  Face Ids
  - 0 -
  3   1
  - 2 -
  """

  def _set_face_nodeid(self, g_cell_nodeid):
    for i in range(0, self.nb_cells):
      self.face_nodeid[i, 0] = np.array([g_cell_nodeid[i, 0], g_cell_nodeid[i, 1]], dtype=np.int32)
      self.face_nodeid[i, 1] = np.array([g_cell_nodeid[i, 1], g_cell_nodeid[i, 2]], dtype=np.int32)
      self.face_nodeid[i, 2] = np.array([g_cell_nodeid[i, 2], g_cell_nodeid[i, 0]], dtype=np.int32)

  def _set_face_measure(self, face_vertices):
    for i in range(self.nb_cells):
      d = face_vertices[i, :, 0] - face_vertices[i, :, 1]
      d = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

      self.faces_measure[i] = d

  def _set_face_center(self, face_vertices):
    for i in range(self.nb_cells):
      center = np.sum(face_vertices[i], axis=1) / 2.0
      self.face_center[i] = center

  def _set_face_normal(self, face_vertices, face_center, tmp_cell_center):
    for i in range(self.nb_cells):
      vertices = face_vertices[i]
      x = vertices[:, 1, 0] - vertices[:, 0, 0]
      y = vertices[:, 1, 1] - vertices[:, 0, 1]
      normal = np.zeros(shape=(3, 2), dtype=self.face_normal.dtype)
      normal[:, 0] = -y
      normal[:, 1] = x

      snorm = tmp_cell_center[i] - face_center[i]
      for j in range(3):
        if (snorm[j][0] * normal[j][0] + snorm[j][1] * normal[j][1]) > 0:
          normal[j] *= -1

      self.face_normal[i] = normal

  def _set_face_ghostcenter(self, ghost_info, face_ghostid):
    for i in range(self.nb_cells):
      for j in range(3):
        if face_ghostid[i, j] != -1:
          self.face_ghostcenter[i][j][:] = ghost_info[face_ghostid[i, j]][0:3]
        else:
          self.face_ghostcenter[i][j][:] = -1

  def _set_face_vertices(self, cell_vertices):
    for i in range(self.nb_cells):
      c_vertices = cell_vertices[i]

      arr = np.array([
        [c_vertices[0], c_vertices[1]],
        [c_vertices[1], c_vertices[2]],
        [c_vertices[2], c_vertices[0]],
      ], dtype=self.faces_vertices.dtype)

      self.faces_vertices[i] = arr

  def _set_g_face_cellid(self):
    width = self.width
    for i in range(self.nb_cells):
      sq_id = i // 2
      if i % 2 == 0:
        arr = np.array([
          [i, -1 if sq_id % width - 1 < 0 else i - 1],
          [i, i + 1],
          [i, -1 if sq_id - width < 0 else i - width * 2 + 1],
        ], dtype=np.int32)
      else:
        arr = np.array([
          [i, i - 1],
          [i, -1 if sq_id + width >= width * width else i + width * 2 - 1],
          [i, -1 if sq_id % width + 1 >= width else i + 1],
        ], dtype=np.int32)
      self.g_face_cellid[i] = arr

  def _set_l_face_cellid(self, l_face_name, g_face_cellid):
      for i in range(self.nb_cells):
        name = l_face_name[i]
        arr = g_face_cellid[i].copy()
        arr[:, 1][name == 10] = -1
        self.l_face_cellid[i] = arr

  def _set_l_and_g_face_name(self, g_face_cellid, cell_which_partition):
    width = self.width
    for i in range(self.nb_cells):
      sq_id = i // 2
      sq_id_x = sq_id // width
      sq_id_y = sq_id % width
      name = np.array([0, 0, 0])
      for j in range(3):
        if sq_id_x == 0 and i % 2 == 0 and j == 2:
          name[j] = 1
        if sq_id_y == width - 1 and i % 2 != 0 and j == 2:
          name[j] = 4
        if sq_id_x == width - 1 and i % 2 != 0 and j == 1:
          name[j] = 2
        if sq_id_y == 0 and i % 2 == 0 and j == 0:
          name[j] = 3

      self.g_face_name[i] = name

      for j in range(3):
        face_cellid = g_face_cellid[i][j]
        if face_cellid[0] != -1 and face_cellid[1] != -1:
          cell_1_partition = cell_which_partition[face_cellid[0]]
          cell_2_partition = cell_which_partition[face_cellid[1]]
          if cell_1_partition != cell_2_partition:
            name[j] = 10

      self.l_face_name[i] = name

  def _set_cell_nf(self, faces_normal):
      self.cell_nf = faces_normal.copy()

# region  ghost node halo

  ###################
  ## Ghost Info
  ###################

  def _set_ghost_info(self, face_nodeid, cell_which_partition, cell_center):
    # Set face_ghostid
    # Set g_node_ghostid
    # Set ghost_info => [center_x, center_y, volume, cell_partition_id, cell_id, cell_face_id(0..3)]

    self.g_node_ghostid[:, -1] = 0

    def add_ghost(ghostcenter, cellid, ghostid, faceid):

      def add_node_ghostid(node_ghostid, ghostid):
        node_ghostid[node_ghostid[-1]] = ghostid
        node_ghostid[-1] += 1

      arr = np.ones(shape=self.ghost_info.shape[1], dtype=self.ghost_info.dtype) * -1

      # face_nodeid[cellid, faceid] = np.sort(face_nodeid[cellid, faceid])[::-1]
      # print(face_nodeid[cellid, faceid])
      # x_1 = g_nodes[face_nodeid[cellid, faceid, 1]]
      # x_2 = g_nodes[face_nodeid[cellid, faceid, 0]]
      # v_1 = cell_center[cellid]
      # gamma = ((v_1[0] - x_2[0]) * (x_1[0] - x_2[0]) + (v_1[1] - x_2[1]) * (x_1[1] - x_2[1])) / (
      #           (x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2)
      gamma = -1

      arr[0:2] = ghostcenter[0:2]
      arr[2] = gamma # gamma
      arr[3] = cell_which_partition[cellid]  # cell partition id
      arr[4] = cellid  # cell id
      arr[5] = faceid  # faceid in the cell (0..3)
      self.ghost_info[ghostid][:] = arr
      # Set face ghost_id
      self.face_ghostid[cellid][faceid] = ghostid
      fnodeid = face_nodeid[cellid][faceid]
      add_node_ghostid(self.g_node_ghostid[fnodeid[0]], ghostid=ghostid)
      add_node_ghostid(self.g_node_ghostid[fnodeid[1]], ghostid=ghostid)



    x_length = 0.333333 * 2
    y_length = 0.166666 * 2
    cmp = 0
    for i in range(self.nb_cells):
      if i % 2 == 0:
        c_center = cell_center[i]

        if (c_center[0] - x_length) < 0:
          ghostcenter = np.array([c_center[0] - x_length, c_center[1]])
          add_ghost(ghostcenter, i, cmp, 2)
          cmp += 1

        if (c_center[1] - y_length) < 0:
          ghostcenter = np.array([c_center[0], c_center[1] - y_length])
          add_ghost(ghostcenter, i, cmp, 0)
          cmp += 1
      else:
        c_center = cell_center[i]

        if (c_center[0] + x_length) > self.WIDTH:
          ghostcenter = np.array([c_center[0] + x_length, c_center[1]])
          add_ghost(ghostcenter, i, cmp, 1)
          cmp += 1

        if (c_center[1] + y_length) > self.HEIGHT:
          ghostcenter = np.array([c_center[0], c_center[1] + y_length])
          add_ghost(ghostcenter, i, cmp, 2)
          cmp += 1

  def _set_l_node_ghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      for i in range(0, 3):
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        not_the_same_partition = (ghost_info[arr][:, 3] != cell_partition_id)
        arr[not_the_same_partition] = -1
        arr = arr[arr != -1]
        self.l_node_ghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.l_node_ghostid[cell_id, i, -1] = len(arr)

  def _set_node_haloghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      for i in range(0, 3):
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        same_partition = (ghost_info[arr][:, 3] == cell_partition_id)
        arr[same_partition] = -1
        arr = arr[arr != -1]
        self.node_haloghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.node_haloghostid[cell_id, i, -1] = len(arr)

  def _set_g_cell_ghostnid(self, g_node_ghostid, g_cell_nodeid):
    for cell_id in range(self.nb_cells):
      res = (g_node_ghostid[g_cell_nodeid[cell_id][0:3]][:, 0:-1]).flatten()
      res = np.unique(res[res != -1])
      self.g_cell_ghostnid[cell_id, 0:len(res)] = res[:]
      self.g_cell_ghostnid[cell_id, -1] = len(res)

  def _set_l_cell_ghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      arr = g_cell_ghostnid[cell_id].copy()
      arr = arr[0:arr[-1]]
      not_the_same_partition = (ghost_info[arr][:, 3] != cell_partition_id) # not the same
      arr[not_the_same_partition] = -1
      arr = arr[arr != -1]
      self.l_cell_ghostnid[cell_id, 0:len(arr)] = arr[:]
      self.l_cell_ghostnid[cell_id, -1] = len(arr)

  def _set_cell_haloghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      arr = g_cell_ghostnid[cell_id].copy()
      arr = arr[0:arr[-1]]
      same_partition = (ghost_info[arr][:, 3] == cell_partition_id) # the same
      arr[same_partition] = -1
      arr = arr[arr != -1]
      self.cell_haloghostnid[cell_id, 0:len(arr)] = arr[:]
      self.cell_haloghostnid[cell_id, -1] = len(arr)

  ###################
  ## Node Info
  ###################

  def _set_g_node_cellid(self, g_cell_nodeid):
    self.g_node_cellid[:, -1] = 0
    for cell_id in range(self.nb_cells):
      for i in range(3):
        node = self.g_node_cellid[g_cell_nodeid[cell_id, i]]
        node[node[-1]] = cell_id
        node[-1] += 1

  def _set_l_node_cellid(self, g_cell_nodeid, g_node_cellid, which_partition):
    for cell_id in range(self.nb_cells):
      for i in range(3):
        node_id = g_cell_nodeid[cell_id, i]
        node_cellid = g_node_cellid[node_id]
        node_cellid = node_cellid[0:node_cellid[-1]]
        node_cellid = node_cellid.copy()
        this_cell_partition = which_partition[cell_id]
        node_cellid[which_partition[node_cellid] != this_cell_partition] = -1
        node_cellid = node_cellid[node_cellid != -1]

        self.l_node_cellid[cell_id][i][0:len(node_cellid)] = node_cellid[:]
        self.l_node_cellid[cell_id][i][-1] = len(node_cellid)

  def _set_node_halonid(self, g_cell_nodeid, g_node_cellid, which_partition):
    # The same as set_l_node_cellid
    for cell_id in range(self.nb_cells):
      for i in range(3):
        node_id = g_cell_nodeid[cell_id, i]
        node_cellid = g_node_cellid[node_id]
        node_cellid = node_cellid[0:node_cellid[-1]]
        node_cellid = node_cellid.copy()
        this_cell_partition = which_partition[cell_id]
        node_cellid[which_partition[node_cellid] == this_cell_partition] = -1 #equal instead of !=
        node_cellid = node_cellid[node_cellid != -1]

        self.node_halonid[cell_id][i][0:len(node_cellid)] = node_cellid[:]
        self.node_halonid[cell_id][i][-1] = len(node_cellid)

  def _set_node_oldname(self):
    width = self.width
    for i in range(self.nb_cells):
      sq_id = i // 2
      sq_id_x = sq_id // width
      sq_id_y = sq_id % width
      name = np.array([0, 0, 0, 0])
      for j in range(0, 4):
        if sq_id_y == 0 and (j == 0 or j ==  1):
          name[j] = 3
        if sq_id_y == width - 1 and (j == 2 or j == 3):
          name[j] = 4
        if sq_id_x == 0 and (j == 0 or j == 3):
          name[j] = 1
        if sq_id_x == width - 1 and (j == 1 or j == 2):
          name[j] = 2
      if i % 2 == 0:
        name = name[[0, 1, 3]]
      else:
        name = name[[3, 1, 2]]

      self.g_node_name[i] = name

  def _set_node_name(self, g_node_name, l_face_name, g_cell_nodeid):
    for i in range(self.nb_cells):
      name = g_node_name[i].copy()

      cell_nodes = g_cell_nodeid[i]
      cell_nodes = cell_nodes[0:cell_nodes[-1]]
      face_nodes = np.array([
          [0, 1],
          [1, 2],
          [2, 0]
      ])
      for face_id in range(3):
        face_name = l_face_name[i][face_id]
        for node_id in range(2):
          l_node_id = face_nodes[face_id][node_id]
          g_node_id = cell_nodes[l_node_id]
          if face_name == 10:
            self.l_node_name[g_node_id] = face_name
          if self.l_node_name[g_node_id] == -1:
            self.l_node_name[g_node_id] = name[l_node_id]




  ###################
  ## Halo Info
  ###################

  def _set_halo_halosint(self, cell_halonid, cell_which_partition):
    res = [[] for i in range(self.nb_partitions)]
    for i in range(len(cell_halonid)):
      item = cell_halonid[i]
      p = cell_which_partition[i]
      if np.any(item != -1):
        res[p].append(i)

    max_len = max(len(subarray) for subarray in res)
    res = [subarray + [-1] * (max_len - len(subarray)) for subarray in res]
    self.halo_halosint = np.array(res, dtype=np.int32)

  def _set_halo_neigh(self, cell_halonid, cell_which_partition):
    haloext = [[] for i in range(self.nb_partitions)]
    for i in range(len(cell_halonid)):
      item = cell_halonid[i]
      p = cell_which_partition[i]
      haloext[p] += list(item[item != -1])

    for p in range(self.nb_partitions):
      for j in range(self.nb_partitions):
        tmp = np.array(haloext[p], dtype=np.int32)
        tmp = np.unique(tmp)
        self.halo_neigh[p][j] = np.sum(cell_which_partition[tmp] == j) #ext that belong to partition j

  def _set_halo_sizehaloghost(self, node_haloghostid, cell_which_partition):
    a = node_haloghostid[:, :, 0:-1]
    haloghost_cells = [[] for i in range(self.nb_partitions)]
    for i in range(len(a)):
      b = a[i][a[i] > 0]
      p = cell_which_partition[i]
      haloghost_cells[p].extend(b)
    for p in range(self.nb_partitions):
      self.halo_sizehaloghost[p] = len(np.unique(haloghost_cells[p]))


  ###################
  ## Cell Info
  ###################

  def _set_g_cell_cellfid(self):
    width = self.width
    for i in range(self.nb_cells):
      sq_id = i // 2
      if i % 2 == 0:
        # all neighboring cells
        arr = np.array([i + 1, i - 1, i - width * 2 + 1], dtype=np.int32)
        # disable non-existing neighboring cells
        if sq_id % width - 1 < 0:
          arr[1] = -1
        if sq_id - width < 0:
          arr[2] = -1
        self.g_cell_cellfid[i][0:arr.shape[0]] = arr
        self.g_cell_cellfid[i][-1] = arr.shape[0]
      else:
        # all neighboring cells
        arr = np.array([i - 1, i + 1, i + width * 2 - 1], dtype=np.int32)
        # disable non-existing neighboring cells
        if sq_id % width + 1 >= width:
          arr[1] = -1
        if sq_id + width >= width * width:
          arr[2] = -1
        self.g_cell_cellfid[i][0:arr.shape[0]] = arr
        self.g_cell_cellfid[i][-1] = arr.shape[0]

  def _set_l_cell_cellfid(self, g_cell_cellfid, cell_which_partition):
    for i in range(self.nb_cells):
      arr = (g_cell_cellfid[i][0:g_cell_cellfid[i][-1]]).copy()
      this_cell_partition = cell_which_partition[i]
      arr[cell_which_partition[arr] != this_cell_partition] = -1
      arr = arr[arr != -1]

      self.l_cell_cellfid[i][0:arr.shape[0]] = arr
      self.l_cell_cellfid[i][-1] = arr.shape[0]

  def _set_g_cell_cellnid(self):
    width = self.width
    for i in range(self.nb_cells):
      sq_id = i // 2

      arr = np.array([
        i - width * 2 - 3,
        i - width * 2 - 2,
        i - width * 2 - 1,
        i - width * 2,
        i - width * 2 + 1,
        i - width * 2 + 2,
        i - 3, #6
        i - 2,
        i - 1, # index 8, neighbor in the same cell
        i + 1,
        i + 2,
        i + width * 2 - 3, #11
        i + width * 2 - 2,
        i + width * 2 - 1,
        i + width * 2,
        i + width * 2 + 1,
        i + width * 2 + 2,
      ], dtype=np.int32)
      if i % 2 == 0:
        arr = arr + 1
        arr[8] = i + 1
        arr[[0, 10, 14, 15, 16]] = -1
      else:
        arr[[0, 1, 2, 6, 16]] = -1
      if sq_id % width - 1 < 0:
        arr[[0, 1, 6, 7, 11, 12]] = -1
      if sq_id % width + 1 >= width:
        arr[[4, 5, 9, 10, 15, 16]] = -1
      if sq_id + width >= width * width:
        arr[[11, 12, 13, 14, 15, 16]] = -1
      if sq_id - width < 0:
        arr[[0, 1, 2, 3, 4, 5]] = -1
      self.g_cell_cellnid[i][0:arr.shape[0]] = arr
      self.g_cell_cellnid[i][-1] = arr.shape[0]

  def _set_l_cell_cellnid(self, g_cell_cellnid):
    for i in range(self.nb_cells):
      arr = (g_cell_cellnid[i][0:g_cell_cellnid[i][-1]]).copy()
      this_cell_partition = self.cell_which_partition[i]
      arr[self.cell_which_partition[arr] != this_cell_partition] = -1
      arr = arr[arr != -1]

      self.l_cell_cellnid[i][0:arr.shape[0]] = arr
      self.l_cell_cellnid[i][-1] = arr.shape[0]

  def _set_cell_vertices(self):
    With = self.WIDTH
    Height = self.HEIGHT
    StepX = With / self.width
    StepY = Height / self.width

    cmp = 0
    for x in np.arange(0, With, StepX):
      for y in np.arange(0, Height, StepY):
        p1 = np.array([x, y], dtype=self.cell_vertices.dtype)
        p2 = np.array([x + StepX, y], dtype=self.cell_vertices.dtype)
        p3 = np.array([x + StepX, y + StepY], dtype=self.cell_vertices.dtype)
        p4 = np.array([x, y + StepY], dtype=self.cell_vertices.dtype)

        t1_points = np.array([p1, p2, p4], dtype=self.cell_vertices.dtype)
        t2_points = np.array([p4, p2, p3], dtype=self.cell_vertices.dtype)

        self.cell_vertices[cmp + 0, :] = t1_points
        self.cell_vertices[cmp + 1, :] = t2_points
        cmp += 2

  def _set_cell_halofid(self, g_cell_cellfid):
    for i in range(self.nb_cells):
      cellfid = (g_cell_cellfid[i][0:g_cell_cellfid[i][-1]]).copy()
      partition_id = self.cell_which_partition[i]
      cellfid_partition_id = self.cell_which_partition[cellfid]
      halofid = cellfid[cellfid_partition_id != partition_id]

      self.cell_halofid[i][:halofid.shape[0]] = halofid

  def _set_cell_halonid(self, g_cell_cellnid):
    for i in range(self.nb_cells):
      cellnid = (g_cell_cellnid[i][0:g_cell_cellnid[i][-1]]).copy()
      partition_id = self.cell_which_partition[i]
      cellnid_partition_id = self.cell_which_partition[cellnid]
      halonid = cellnid[cellnid_partition_id != partition_id]

      self.cell_halonid[i][:halonid.shape[0]] = halonid

  def _set_cell_center(self, cell_vertices):
    for i in range(self.nb_cells):
      points = cell_vertices[i]

      # Center
      center = np.sum(points, axis=0) / 3.0
      self.cell_center[i, :] = center

  def _set_cell_area(self):
    area = self.x_length * self.y_length * 0.5
    self.cell_area[:] = area

  def _set_cell_which_partition(self):
    nb_partitions = self.nb_partitions
    d_loctoglob = self.d_cell_loctoglob

    for p in range(nb_partitions):
      loctoglob = d_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        self.cell_which_partition[global_index] = p

# endregion


  def init(self):
    self._set_cell_which_partition()

    # Cell
    self._set_cell_vertices()
    self._set_cell_center(self.cell_vertices)
    self._set_cell_area()
    self._set_g_cell_cellfid()
    self._set_g_cell_cellnid()
    self._set_l_cell_cellfid(self.g_cell_cellfid, self.cell_which_partition)
    self._set_l_cell_cellnid(self.g_cell_cellnid)
    self._set_cell_halofid(self.g_cell_cellfid)
    self._set_cell_halonid(self.g_cell_cellnid)

    # ## Face
    self._set_face_vertices(self.cell_vertices)
    self._set_face_measure(self.faces_vertices)
    self._set_face_center(self.faces_vertices)
    self._set_g_face_cellid()
    self._set_l_and_g_face_name(self.g_face_cellid, self.cell_which_partition)
    self._set_l_face_cellid(self.l_face_name, self.g_face_cellid)
    self._set_face_normal(self.faces_vertices, self.face_center, self.cell_center)
    self._set_cell_nf(self.face_normal)

    ## ghost
    self._set_face_nodeid(self.g_cell_nodeid)
    self._set_g_node_cellid(self.g_cell_nodeid)
    self._set_ghost_info(self.face_nodeid, self.cell_which_partition, self.cell_center) #ghost_info, face_ghostid, node_ghostid
    self._set_g_cell_ghostnid(self.g_node_ghostid, self.g_cell_nodeid)
    self._set_l_cell_ghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_cell_haloghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_face_ghostcenter(self.ghost_info, self.face_ghostid)


    # ## Node
    self._set_l_node_cellid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    self._set_node_halonid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    self._set_node_oldname()
    self._set_node_name(self.g_node_name, self.l_face_name, self.g_cell_nodeid)
    self._set_l_node_ghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    self._set_node_haloghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)

    # ## Halo
    self._set_halo_halosint(self.cell_halonid, self.cell_which_partition)
    self._set_halo_neigh(self.cell_halonid, self.cell_which_partition)
    self._set_halo_sizehaloghost(self.node_haloghostid, self.cell_which_partition)

################
## Usage
################

# domain_tables = DomainTables(nb_partitions=4, mesh_name=mesh_name, float_precision=float_precision, dim=dim,
#                              create_par_fun=create_partitions)
# unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim,
#                               create_par_fun=create_partitions)
#
# d_cell_loctoglob = domain_tables.d_cell_loctoglob
# g_cell_nodeid = unified_domain.d_cell_nodeid[0]
# a_test = TestTablesRect2D(np.float32, d_cell_loctoglob, g_cell_nodeid)
# a_test.init()

