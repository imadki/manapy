import numpy as np

class TablesTestHexa3D:
  def __init__(self, float_precision, d_cell_loctoglob, g_cell_nodeid):
    """
    d_cell_loctoglob: loctoglob of the local domains
    g_cell_nodeid: cell_nodeid of the global domain

    y -> z -> x
    unit vector [1, 0.5, 1.5]
    Axes:
         ↑ Z
         |
         |
         •------→ X
        /
       /
      Y

       5-------4
      /|      /|  Face 1 [0, 1, 2, 3] 3
     1-------0 |  Face 2 [4, 7, 6, 5] 4
     | |     | |  Face 3 [0, 4, 7, 3] 1
     | 6-----|-7  Face 4 [1, 5, 6, 2] 2
     |/      |/   Face 5 [0, 4, 5, 1] 6
     2-------3    Face 6 [3, 7, 6, 2] 5

    Each hexahedron can be divided into six tetrahedrons

       5-------4
      /|      /|  Tetra 1 [0, 1, 3, 4]
     1-------0 |  Tetra 2 [1, 3, 4, 5]
     | |     | |  Tetra 3 [4, 5, 3, 7]
     | 6-----|-7  Tetra 4 [1, 3, 5, 2]
     |/      |/   Tetra 5 [3, 7, 5, 2]
     2-------3    Tetra 6 [5, 7, 6, 2]

    """

    if float_precision == 'float32':
      float_precision = np.float32
    elif float_precision == 'float64':
      float_precision = np.float64
    else:
      raise ValueError('float_precision must be "float32" or "float64"')

    width = 10

    self.width = width # number of rectangles along the x-axis, y-axis and z-axis
    self.WIDTH = 10.0
    self.HEIGHT = 5.0
    self.DEPTH = 15.0
    self.nb_faces = 3300
    self.nb_cells = self.width * self.width * self.width
    self.nb_nodes = 1331
    self.nb_ghosts = 600

    self.x_length = self.WIDTH / self.width
    self.y_length = self.HEIGHT / self.width
    self.z_length = self.DEPTH / self.width
    self.nb_partitions = len(d_cell_loctoglob)

    self.d_cell_loctoglob = d_cell_loctoglob
    self.g_cell_nodeid = g_cell_nodeid

    self.cell_vertices = np.zeros(shape=(self.nb_cells, 8, 3), dtype=float_precision)
    self.cell_center = np.zeros(shape=(self.nb_cells, 3), dtype=float_precision)
    self.cell_area = np.zeros(shape=(self.nb_cells), dtype=float_precision)
    self.cell_which_partition = np.ones(shape=(self.nb_cells), dtype=np.int32) * -1
    self.cell_halonid = np.ones(shape=(self.nb_cells, 26), dtype=np.int32) * -1
    self.cell_halofid = np.ones(shape=(self.nb_cells, 6), dtype=np.int32) * -1
    self.g_cell_cellnid = np.ones(shape=(self.nb_cells, 27), dtype=np.int32) * -1
    self.l_cell_cellnid = np.ones(shape=(self.nb_cells, 27), dtype=np.int32) * -1
    self.g_cell_cellfid = np.ones(shape=(self.nb_cells, 7), dtype=np.int32) * -1
    self.l_cell_cellfid = np.ones(shape=(self.nb_cells, 7), dtype=np.int32) * -1
    self.cell_nf = np.zeros(shape=(self.nb_cells, 6, 3), dtype=float_precision)

    self.faces_measure = np.zeros(shape=(self.nb_cells, 6), dtype=float_precision)
    self.face_center = np.zeros(shape=(self.nb_cells, 6, 3), dtype=float_precision)
    self.face_normal = np.zeros(shape=(self.nb_cells, 6, 3), dtype=float_precision)
    self.faces_vertices = np.zeros(shape=(self.nb_cells, 6, 4, 3), dtype=float_precision)
    self.g_face_name = np.ones(shape=(self.nb_cells, 6), dtype=np.int32) * -1
    self.l_face_name = np.ones(shape=(self.nb_cells, 6), dtype=np.int32) * -1
    self.g_face_cellid = np.ones(shape=(self.nb_cells, 6, 2), dtype=np.int32) * -1
    self.l_face_cellid = np.ones(shape=(self.nb_cells, 6, 2), dtype=np.int32) * -1
    self.face_nodeid = np.ones(shape=(self.nb_cells, 6, 4), dtype=np.int32) * -1

    self.g_node_cellid = np.ones(shape=(self.nb_nodes, 9), dtype=np.int32) * -1
    self.l_node_cellid = np.ones(shape=(self.nb_cells, 8, 9), dtype=np.int32) * -1
    self.node_halonid = np.ones(shape=(self.nb_cells, 8, 9), dtype=np.int32) * -1
    self.g_node_name = np.ones(shape=(self.nb_cells, 8), dtype=np.int32) * -1
    self.l_node_name = np.ones(shape=self.nb_nodes, dtype=np.int32) * -1

    self.ghost_info = np.ones(shape=(self.nb_ghosts, 7), dtype=float_precision) * -1
    self.g_cell_ghostnid = np.ones(shape=(self.nb_cells, 12+1), dtype=np.int32) * -1
    self.l_cell_ghostnid = np.ones(shape=(self.nb_cells, 12+1), dtype=np.int32) * -1
    self.cell_haloghostnid = np.ones(shape=(self.nb_cells, 12+1), dtype=np.int32) * -1
    self.face_ghostid = np.ones(shape=(self.nb_cells, 6), dtype=np.int32) * -1
    self.face_ghostcenter = np.zeros(shape=(self.nb_cells, 6, 4), dtype=float_precision)
    self.g_node_ghostid = np.ones(shape=(self.nb_nodes, 5), dtype=np.int32) * -1
    self.l_node_ghostid = np.ones(shape=(self.nb_cells, 8, 5), dtype=np.int32) * -1
    self.node_haloghostid = np.ones(shape=(self.nb_cells, 8, 5), dtype=np.int32) * -1

    self.halo_halosint = np.array([], np.int32)
    self.halo_neigh = np.zeros(shape=(self.nb_partitions, self.nb_partitions), dtype=np.int32)
    self.halo_sizehaloghost = np.zeros(shape=(self.nb_partitions), dtype=np.int32)




  def _set_face_nodeid(self, g_cell_nodeid):
    for i in range(0, self.nb_cells):
      cell_nodes = g_cell_nodeid[i]
      self.face_nodeid[i, 0] = np.array([cell_nodes[[0, 1, 2, 3]]], dtype=np.int32)
      self.face_nodeid[i, 1] = np.array([cell_nodes[[4, 7, 5, 6]]], dtype=np.int32)
      self.face_nodeid[i, 2] = np.array([cell_nodes[[0, 4, 7, 3]]], dtype=np.int32)
      self.face_nodeid[i, 3] = np.array([cell_nodes[[1, 5, 6, 2]]], dtype=np.int32)
      self.face_nodeid[i, 4] = np.array([cell_nodes[[0, 4, 5, 1]]], dtype=np.int32)
      self.face_nodeid[i, 5] = np.array([cell_nodes[[3, 7, 6, 2]]], dtype=np.int32)

  def _set_face_measure(self):
    x = self.x_length
    y = self.y_length
    z = self.z_length
    for i in range(self.nb_cells):
      arr = np.array([
        x * z,
        x * z,
        y * z,
        y * z,
        x * y,
        x * y
      ])
      self.faces_measure[i] = arr

  def _set_face_center(self, face_vertices):
    for i in range(self.nb_cells):
      center = np.sum(face_vertices[i], axis=1) / 4.0
      self.face_center[i] = center

  def _set_face_normal(self, face_vertices, face_center, tmp_cell_center):

    def normal_from_triangle(points: 'float[:]'):
      """
      Computes the normal vector of a triangle surface defined by 3 points.
      Returns a 3D normal vector (nx, ny, nz).
      """
      p1 = points[0]
      p2 = points[1]
      p3 = points[2]

      v1 = np.array(p2) - np.array(p1)
      v2 = np.array(p3) - np.array(p1)
      normal = np.cross(v1, v2)

      return normal

    def normal_from_rectangle(points: 'float[:]'):
      """
      Computes the average normal of a rectangle surface (assumed planar).
      Uses two triangles: (p1, p2, p3) and (p1, p3, p4).
      Returns the average unit normal vector.
      """
      p1 = points[0]
      p2 = points[1]
      p3 = points[2]
      p4 = points[3]

      n1 = np.array(normal_from_triangle(np.array([p1, p2, p3])))
      n2 = np.array(normal_from_triangle(np.array([p1, p3, p4])))
      normal = (n1 + n2) / 2

      return normal

    for i in range(self.nb_cells):
      normal = np.zeros(shape=self.face_normal[0].shape, dtype=self.face_normal.dtype)

      for j in range(len(face_vertices[i])):
        vertices = face_vertices[i, j]
        normal[j] = normal_from_rectangle(vertices)
        snorm = tmp_cell_center[i] - face_center[i, j]
        if (np.dot(normal[j], snorm)) > 0:
          normal[j] *= -1

      self.face_normal[i] = normal

  def _set_face_ghostcenter(self, ghost_info, face_ghostid):
    # self.face_ghostcenter => [center_x center_y center_z gamma]
    for i in range(self.nb_cells):
      for j in range(6):
        if face_ghostid[i, j] != -1:
          self.face_ghostcenter[i][j][:] = ghost_info[face_ghostid[i, j]][0:4]
        else:
          self.face_ghostcenter[i][j][:] = -1

  def _set_face_vertices(self, cell_vertices):
    for i in range(self.nb_cells):
      cellv = cell_vertices[i]
      a = np.array([
        cellv[[0, 1, 2, 3]],
        cellv[[4, 7, 6, 5]],
        cellv[[0, 4, 7, 3]],
        cellv[[1, 5, 6, 2]],
        cellv[[0, 4, 5, 1]],
        cellv[[3, 7, 6, 2]],
      ])

      self.faces_vertices[i] = a

  def _set_g_face_cellid(self):
    for i in range(self.nb_cells):
      width = self.width
      size = self.width * self.width

      arr = np.array([
        [i, -1 if i % width - 1 < 0 else i - 1],
        [i, -1 if i % width + 1 >= width else i + 1],
        [i, -1 if i - size < 0 else i - size],
        [i, -1 if i + size >= width * width * width else i + size],
        [i, -1 if i % size - width < 0 else i - width],
        [i, -1 if i % size + width >= size else i + width],
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
    for sq_id in range(self.nb_cells):
      sq_id_x = sq_id // (width * width)
      sq_id_y = sq_id % width
      sq_id_z = (sq_id // width) % width
      name = np.array([0, 0, 0, 0, 0, 0])
      for i in range(0, 6):
        if sq_id_y == width - 1 and i == 1:
          name[i] = 4
        if sq_id_y == 0 and i == 0:
          name[i] = 3
        if sq_id_x == width - 1 and i == 3:
          name[i] = 2
        if sq_id_x == 0 and i == 2:
          name[i] = 1
        if sq_id_z == width - 1 and i == 5:
          name[i] = 5
        if sq_id_z == 0 and i == 4:
          name[i] = 6

      self.g_face_name[sq_id] = name

      for i in range(0, 6):
        face_cellid = g_face_cellid[sq_id][i]
        if face_cellid[0] != -1 and face_cellid[1] != -1:
          cell_1_partition = cell_which_partition[face_cellid[0]]
          cell_2_partition = cell_which_partition[face_cellid[1]]
          if cell_1_partition != cell_2_partition:
            name[i] = 10

      self.l_face_name[sq_id] = name

  def _set_cell_nf(self, faces_normal):
    self.cell_nf = faces_normal.copy()


  ###################
  ## Ghost Info
  ###################

  def _set_ghost_info(self, face_nodeid, cell_which_partition, cell_center, g_face_name, face_center):
    # Set face_ghostid
    # Set g_node_ghostid
    # Set ghost_info => [center_x, center_y, center_z, volume, cell_partition_id, cell_id, cell_face_id(0..6)]
    self.g_node_ghostid[:, -1] = 0
    def add_ghost(ghostcenter, cellid, ghostid, faceid):

      def add_node_ghostid(node_ghostid, ghostid):
        node_ghostid[node_ghostid[-1]] = ghostid
        node_ghostid[-1] += 1

      arr = np.ones(shape=self.ghost_info.shape[1], dtype=self.ghost_info.dtype) * -1
      gamma = -1

      arr[0:3] = ghostcenter[0:3]
      arr[3] = gamma # gamma
      arr[4] = cell_which_partition[cellid]  # cell partition id
      arr[5] = cellid  # cell id
      arr[6] = faceid  # faceid in the cell (0..6)
      self.ghost_info[ghostid][:] = arr
      # Set face ghost_id
      self.face_ghostid[cellid, faceid] = ghostid
      fnodeid = face_nodeid[cellid, faceid]
      for i in range(4): #each face has 4 nodes
        add_node_ghostid(self.g_node_ghostid[fnodeid[i]], ghostid=ghostid)

    cmp = 0
    for i in range(self.nb_cells):
      c_center = cell_center[i]
      for j in range(6): #each cell has 6 faces
        f_center = face_center[i, j]
        f_oldname = g_face_name[i, j]
        if f_oldname != 0:
          diff = f_center - c_center
          ghostcenter = f_center + diff
          add_ghost(ghostcenter, i, cmp, j)
          cmp += 1


  def _set_l_node_ghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      nb_nodes = 8
      for i in range(0, nb_nodes): # number of nodes
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        ghost_partition_id = ghost_info[arr][:, 4]
        not_the_same_partition = (ghost_partition_id != cell_partition_id)
        arr[not_the_same_partition] = -1
        arr = arr[arr != -1]
        self.l_node_ghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.l_node_ghostid[cell_id, i, -1] = len(arr)

  def _set_node_haloghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      nb_nodes = 8
      for i in range(0, nb_nodes):
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        ghost_partition_id = ghost_info[arr][:, 4]
        same_partition = (ghost_partition_id == cell_partition_id)
        arr[same_partition] = -1
        arr = arr[arr != -1]
        self.node_haloghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.node_haloghostid[cell_id, i, -1] = len(arr)

  def _set_g_cell_ghostnid(self, g_node_ghostid, g_cell_nodeid):
    for cell_id in range(self.nb_cells):
      nb_nodes = 8
      res = (g_node_ghostid[g_cell_nodeid[cell_id][0:nb_nodes]][:, 0:-1]).flatten()
      res = np.unique(res[res != -1])
      self.g_cell_ghostnid[cell_id, 0:len(res)] = res[:]
      self.g_cell_ghostnid[cell_id, -1] = len(res)

  def _set_l_cell_ghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      arr = g_cell_ghostnid[cell_id].copy()
      arr = arr[0:arr[-1]]
      ghost_partition_id = ghost_info[arr][:, 4]
      not_the_same_partition = (ghost_partition_id != cell_partition_id) # not the same
      arr[not_the_same_partition] = -1
      arr = arr[arr != -1]
      self.l_cell_ghostnid[cell_id, 0:len(arr)] = arr[:]
      self.l_cell_ghostnid[cell_id, -1] = len(arr)

  def _set_cell_haloghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      arr = g_cell_ghostnid[cell_id].copy()
      arr = arr[0:arr[-1]]
      ghost_partition_id = ghost_info[arr][:, 4]
      same_partition = (ghost_partition_id == cell_partition_id) # the same
      arr[same_partition] = -1
      arr = arr[arr != -1]
      self.cell_haloghostnid[cell_id, 0:len(arr)] = arr[:]
      self.cell_haloghostnid[cell_id, -1] = len(arr)

  ###################
  ## Node Info
  ###################

  # TODO TEST g_cell_nodeid
  def _set_g_node_cellid(self, g_cell_nodeid):
    self.g_node_cellid[:, -1] = 0
    for cell_id in range(self.nb_cells):
      for i in range(8):
        node = self.g_node_cellid[g_cell_nodeid[cell_id, i]]
        node[node[-1]] = cell_id
        node[-1] += 1

  def _set_l_node_cellid(self, g_cell_nodeid, g_node_cellid, which_partition):
    for cell_id in range(self.nb_cells):
      for i in range(8):
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
      for i in range(8):
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
    for sq_id in range(self.nb_cells):
      sq_id_x = sq_id // (width * width)
      sq_id_y = sq_id % width
      sq_id_z = (sq_id // width) % width
      name = np.array([0, 0, 0, 0, 0, 0, 0, 0])
      for i in range(0, 8):
        if sq_id_z == width - 1 and i in [3, 7, 6, 2]:
          name[i] = 5
        if sq_id_z == 0 and i in [0, 4, 5, 1]:
          name[i] = 6
        if sq_id_y == width - 1 and i in [4, 7, 5, 6]:
          name[i] = 4
        if sq_id_y == 0 and i in [0, 1, 2, 3]:
          name[i] = 3
        if sq_id_x == 0 and i in [0, 4, 7, 3]:
          name[i] = 1
        if sq_id_x == width - 1 and i in [1, 5, 6, 2]:
          name[i] = 2

      self.g_node_name[sq_id] = name

  def _set_node_name(self, g_node_name, l_face_name, g_cell_nodeid):
    for sq_id in range(self.nb_cells):
      name = g_node_name[sq_id].copy()

      cell_nodes = g_cell_nodeid[sq_id]
      cell_nodes = cell_nodes[0:cell_nodes[-1]]
      face_nodes = np.array([
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 7, 3],
        [1, 5, 6, 2],
        [0, 4, 5, 1],
        [3, 7, 6, 2],
      ])
      for face_id in range(0, 6):
        face_name = l_face_name[sq_id][face_id]
        for node_id in range(4):
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


  def _set_halo_sizehaloghost(self, node_haloghostid, cell_which_partition, g_cell_nodeid):
    nb_partitions = self.nb_partitions

    haloghost_cells = [[] for i in range(nb_partitions)]
    for i in range(len(g_cell_nodeid)):
      p = cell_which_partition[i]
      for j in range(g_cell_nodeid[i, -1]):
        a = node_haloghostid[i, j]
        a = a[0:a[-1]]
        node_id = g_cell_nodeid[i, j]
        for item in a:
          haloghost_cells[p].append([node_id, item])

    for p in range(nb_partitions):
      tmp = np.array(haloghost_cells[p])
      tmp = np.unique(tmp, axis=0)
      self.halo_sizehaloghost[p] = len(tmp)

  ###################
  ## Cell Info
  ###################

  def _set_g_cell_cellfid(self):
    width = self.width
    size = self.width * self.width
    for i in range(self.nb_cells):
      arr = np.array([
        i - size, #0
        i - width, #1
        i - 1, #2
        i + 1, #3
        i + width, #4
        i + size, #5
      ], dtype=np.int32)
      if i % width - 1 < 0:
        arr[2] = -1
      if i % width + 1 >= width:
        arr[3] = -1
      if i % size - width < 0:
        arr[1] = -1
      if i % size + width >= size:
        arr[4] = -1
      if i - size < 0:
        arr[0] = -1
      if i + size >= width * width * width:
        arr[5] = -1

      arr = arr[arr != -1]

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
    size = self.width * self.width
    for i in range(self.nb_cells):
      arr = np.array([
        i - size - width - 1,  # 0
        i - size - width,      # 1
        i - size - width + 1,  # 2
        i - size - 1,          # 3
        i - size,              # 4
        i - size + 1,          # 5
        i - size + width - 1,  # 6
        i - size + width,      # 7
        i - size + width + 1,  # 8
        i - width - 1,         # 9
        i - width,             # 10
        i - width + 1,         # 11
        i - 1,                 # 12
        i + 1,                 # 13
        i + width - 1,         # 14
        i + width,             # 15
        i + width + 1,         # 16
        i + size - width - 1,  # 17
        i + size - width,      # 18
        i + size - width + 1,  # 19
        i + size - 1,          # 20
        i + size,              # 21
        i + size + 1,          # 22
        i + size + width - 1,  # 23
        i + size + width,      # 24
        i + size + width + 1   # 25
      ], dtype=np.int32)
      if i % width - 1 < 0:
        arr[[0, 3, 6, 9, 12, 14, 17, 20, 23]] = -1
      if i % width + 1 >= width:
        arr[[2, 5, 8, 11, 13, 16, 19, 22, 25]] = -1
      if i % size - width < 0:
        arr[[0, 1, 2, 9, 10, 11, 17, 18, 19]] = -1
      if i % size + width >= size:
        arr[[6, 7, 8, 14, 15, 16, 23, 24, 25]] = -1
      if i - size < 0:
        arr[[0, 1, 2, 3, 4, 5, 6, 7, 8]] = -1
      if i + size >= width * width * width:
        arr[[17, 18, 19, 20, 21, 22, 23, 24, 25]] = -1

      arr = arr[arr != -1]

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
    Width = self.WIDTH
    Height = self.HEIGHT
    Depth = self.DEPTH
    StepX = Width / self.width
    StepY = Height / self.width
    StepZ = Depth / self.width

    cmp = 0
    for x in np.arange(0.0, Width, StepX):
      for z in np.arange(0.0, Depth, StepZ):
        for y in np.arange(Height, 0.0, -StepY):
          points = np.array([
            [x, y, z],
            [x + StepX, y, z],
            [x + StepX, y, z + StepZ],
            [x, y, z + StepZ],

            [x, y - StepY, z],
            [x + StepX, y - StepY, z],
            [x + StepX, y - StepY, z + StepZ],
            [x, y - StepY, z + StepZ],
          ], dtype=self.cell_vertices.dtype)

          self.cell_vertices[cmp] = points
          cmp += 1

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
      center = np.sum(points, axis=0) / 8.0
      self.cell_center[i, :] = center

  def _set_cell_area(self):
    area = self.x_length * self.y_length * self.z_length
    self.cell_area[:] = area

  def _set_cell_which_partition(self):
    nb_partitions = self.nb_partitions
    d_loctoglob = self.d_cell_loctoglob

    for p in range(nb_partitions):
      loctoglob = d_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        self.cell_which_partition[global_index] = p

  def init(self):
    self._set_cell_which_partition()

    ## Cell
    self._set_cell_vertices()
    self._set_cell_center(self.cell_vertices)
    self._set_cell_area()
    self._set_g_cell_cellfid()
    self._set_g_cell_cellnid()
    self._set_l_cell_cellfid(self.g_cell_cellfid, self.cell_which_partition)
    self._set_l_cell_cellnid(self.g_cell_cellnid)
    self._set_cell_halofid(self.g_cell_cellfid)
    self._set_cell_halonid(self.g_cell_cellnid)

    ## Face
    self._set_face_measure()
    self._set_face_vertices(self.cell_vertices)
    self._set_face_center(self.faces_vertices)
    self._set_g_face_cellid()
    self._set_l_and_g_face_name(self.g_face_cellid, self.cell_which_partition)
    self._set_l_face_cellid(self.l_face_name, self.g_face_cellid)
    self._set_face_normal(self.faces_vertices, self.face_center, self.cell_center)
    self._set_cell_nf(self.face_normal)

    ## Node
    self._set_g_node_cellid(self.g_cell_nodeid)
    self._set_l_node_cellid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    self._set_node_halonid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    self._set_node_oldname()
    self._set_node_name(self.g_node_name, self.l_face_name, self.g_cell_nodeid)

    ## ghostid
    self._set_face_nodeid(self.g_cell_nodeid)
    self._set_ghost_info(self.face_nodeid, self.cell_which_partition, self.cell_center, self.g_face_name, self.face_center) #ghost_info, face_ghostid, node_ghostid
    self._set_g_cell_ghostnid(self.g_node_ghostid, self.g_cell_nodeid)
    self._set_l_cell_ghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_cell_haloghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_face_ghostcenter(self.ghost_info, self.face_ghostid)
    self._set_l_node_ghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    self._set_node_haloghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)

    ## Halo
    self._set_halo_halosint(self.cell_halonid, self.cell_which_partition)
    self._set_halo_neigh(self.cell_halonid, self.cell_which_partition)
    self._set_halo_sizehaloghost(self.node_haloghostid, self.cell_which_partition, self.g_cell_nodeid)

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

