import numpy as np

class TablesTestTetra3D:
  def __init__(self, float_precision, d_cell_loctoglob, g_cell_nodeid):
    """
    d_cell_loctoglob: loctoglob of the local domains
    g_cell_nodeid: cell_nodeid of the global domain

    unit vector [1, 0.5, 1.5]
    Axes:
             ↑ Z
             |
             |
             •------→ X
            /
           /
          Y

         0
        /|\        Face 1 [0, 1, 2]
       / | \       Face 2 [0, 1, 3]
      /  |  \      Face 3 [0, 2, 3]
     1---|---2     Face 4 [1, 2, 3]
      \  |  /
       \ | /
        \|/
         3

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
    self.nb_faces = 12600
    self.nb_cells = self.width * self.width * self.width *  6
    self.nb_nodes = 1331
    self.nb_ghosts = 40 #TODO

    self.x_length = self.WIDTH / self.width
    self.y_length = self.HEIGHT / self.width
    self.z_length = self.DEPTH / self.width
    self.nb_partitions = len(d_cell_loctoglob)

    self.d_cell_loctoglob = d_cell_loctoglob
    self.g_cell_nodeid = g_cell_nodeid

    self.cell_vertices = np.zeros(shape=(self.nb_cells, 4, 3), dtype=float_precision)
    self.cell_center = np.zeros(shape=(self.nb_cells, 3), dtype=float_precision)
    self.cell_area = np.zeros(shape=(self.nb_cells), dtype=float_precision)
    self.cell_which_partition = np.ones(shape=(self.nb_cells), dtype=np.int32) * -1
    # self.cell_halonid = np.ones(shape=(self.nb_cells, 26), dtype=np.int32) * -1
    # self.cell_halofid = np.ones(shape=(self.nb_cells, 6), dtype=np.int32) * -1
    # self.g_cell_cellnid = np.ones(shape=(self.nb_cells, 27), dtype=np.int32) * -1
    # self.l_cell_cellnid = np.ones(shape=(self.nb_cells, 27), dtype=np.int32) * -1
    self.g_cell_cellfid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.l_cell_cellfid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.cell_nf = np.zeros(shape=(self.nb_cells, 4, 2), dtype=float_precision)

    self.faces_measure = np.zeros(shape=(self.nb_cells, 4), dtype=float_precision)
    # self.face_center = np.zeros(shape=(self.nb_cells, 4, 2), dtype=float_precision)
    # self.face_normal = np.zeros(shape=(self.nb_cells, 4, 2), dtype=float_precision)
    self.faces_vertices = np.zeros(shape=(self.nb_cells, 4, 2, 2), dtype=float_precision)
    # self.face_ghostcenter = np.zeros(shape=(self.nb_cells, 4, 3), dtype=float_precision)
    # self.g_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.l_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.g_face_cellid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1
    # self.l_face_cellid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1
    # self.face_nodeid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1
    #
    # self.g_node_cellid = np.ones(shape=(self.nb_nodes, 5), dtype=np.int32) * -1
    # self.l_node_cellid = np.ones(shape=(self.nb_cells, 4, 5), dtype=np.int32) * -1
    # self.node_halonid = np.ones(shape=(self.nb_cells, 4, 5), dtype=np.int32) * -1
    # self.g_node_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.l_node_name = np.ones(shape=self.nb_nodes, dtype=np.int32) * -1
    #
    # self.ghost_info = np.ones(shape=(self.nb_ghosts, 6), dtype=float_precision) * -1
    # self.g_cell_ghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    # self.l_cell_ghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    # self.cell_haloghostnid = np.ones(shape=(self.nb_cells, 5), dtype=np.int32) * -1
    # self.face_ghostid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    # self.g_node_ghostid = np.ones(shape=(self.nb_nodes, 3), dtype=np.int32) * -1
    # self.l_node_ghostid = np.ones(shape=(self.nb_cells, 4, 3), dtype=np.int32) * -1
    # self.node_haloghostid = np.ones(shape=(self.nb_cells, 4, 3), dtype=np.int32) * -1
    #
    # self.halo_halosint = np.array([], np.int32)
    # self.halo_neigh = np.zeros(shape=(self.nb_partitions, self.nb_partitions), dtype=np.int32)
    # self.halo_sizehaloghost = np.zeros(shape=(self.nb_partitions), dtype=np.int32)




  def _set_face_nodeid(self, g_cell_nodeid):
    pass

  def _set_face_measure(self, cell_vertices):
    def get_triangle_area(points: np.ndarray) -> float:
      a = points[1] - points[0]
      b = points[2] - points[0]

      # Cross product and area
      cross = np.cross(a, b)
      area = 0.5 * np.linalg.norm(cross)
      return area

    for i in range(self.nb_cells):
      cv = cell_vertices[i]
      face_v = np.array([
        [cv[0], cv[1], cv[2]],
        [cv[0], cv[1], cv[3]],
        [cv[0], cv[2], cv[3]],
        [cv[1], cv[2], cv[3]],
      ])

      measures = np.array([
        get_triangle_area(face_v[0]),
        get_triangle_area(face_v[1]),
        get_triangle_area(face_v[2]),
        get_triangle_area(face_v[3]),
      ])

      self.faces_measure[i] = measures

  def _set_face_center(self, face_vertices):
    pass

  def _set_face_normal(self, face_vertices, face_center, tmp_cell_center):
    pass

  def _set_face_ghostcenter(self, ghost_info, face_ghostid):
    pass

  def _set_face_vertices(self, cell_vertices):
    pass

  def _set_g_face_cellid(self):
    pass

  def _set_l_face_cellid(self, l_face_name, g_face_cellid):
    pass

  def _set_l_and_g_face_name(self, g_face_cellid, cell_which_partition):
    pass

  def _set_cell_nf(self, faces_normal):
    pass


  ###################
  ## Ghost Info
  ###################

  def _set_ghost_info(self, face_nodeid, cell_which_partition):
    pass

  def _set_l_node_ghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    pass

  def _set_node_haloghostid(self, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    pass

  def _set_g_cell_ghostnid(self, g_node_ghostid, g_cell_nodeid):
    pass

  def _set_l_cell_ghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    pass

  def _set_cell_haloghostnid(self, g_cell_ghostnid, ghost_info, cell_which_partition):
    pass

  ###################
  ## Node Info
  ###################

  def _set_g_node_cellid(self, g_cell_nodeid):
    pass

  def _set_l_node_cellid(self, g_cell_nodeid, g_node_cellid, which_partition):
    pass

  def _set_node_halonid(self, g_cell_nodeid, g_node_cellid, which_partition):
    pass

  def _set_node_oldname(self):
    pass

  def _set_node_name(self, g_node_name, l_face_name, g_cell_nodeid):
    pass


  ###################
  ## Halo Info
  ###################

  def _set_halo_halosint(self, cell_halonid, cell_which_partition):
    pass

  def _set_halo_neigh(self, cell_halonid, cell_which_partition):
    pass

  def _set_halo_sizehaloghost(self, node_haloghostid, cell_which_partition):
    pass

  ###################
  ## Cell Info
  ###################

  def _set_g_cell_cellfid(self):
    pass

  def _set_l_cell_cellfid(self, g_cell_cellfid, cell_which_partition):
    pass

  def _set_g_cell_cellnid(self):
    pass

  def _set_l_cell_cellnid(self, g_cell_cellnid):
    pass

  def _set_cell_vertices(self):
    With = self.WIDTH
    Height = self.HEIGHT
    Depth = self.DEPTH
    StepX = With / self.width
    StepY = Height / self.width
    StepZ = Depth / self.width

    cmp = 0
    for x in np.arange(With, 0.0, -StepX):
      for z in np.arange(Depth, 0.0, -StepZ):
        for y in np.arange(Height, 0.0, -StepY):
          points = np.array([
            [x, y, z],
            [x - StepX, y, z],
            [x - StepX, y, z - StepZ],
            [x, y, z - StepZ],

            [x, y - StepY, z],
            [x - StepX, y - StepY, z],
            [x - StepX, y - StepY, z - StepZ],
            [x, y - StepY, z - StepZ],
          ], dtype=self.cell_vertices.dtype)

          tetrahedrons = np.array([
            [points[0], points[1], points[3], points[4]],
            [points[1], points[3], points[4], points[5]],
            [points[4], points[5], points[3], points[7]],
            [points[1], points[3], points[5], points[2]],
            [points[3], points[7], points[5], points[2]],
            [points[5], points[7], points[6], points[2]],
          ])

          self.cell_vertices[cmp + 0] = tetrahedrons[0]
          self.cell_vertices[cmp + 1] = tetrahedrons[1]
          self.cell_vertices[cmp + 2] = tetrahedrons[2]
          self.cell_vertices[cmp + 3] = tetrahedrons[3]
          self.cell_vertices[cmp + 4] = tetrahedrons[4]
          self.cell_vertices[cmp + 5] = tetrahedrons[5]

          cmp += 6

  def _set_cell_halofid(self, g_cell_cellfid):
    pass

  def _set_cell_halonid(self, g_cell_cellnid):
    pass

  def _set_cell_center(self, cell_vertices):
    for i in range(self.nb_cells):
      points = cell_vertices[i]

      # Center
      center = np.sum(points, axis=0) / 4.0
      self.cell_center[i, :] = center

  def _set_cell_area(self, cell_vertices):
    for i in range(self.nb_cells):
      def tetrahedron_volume(points: 'float[:, :]'):
        a, b, c, d = points
        matrix = np.array([b - a, c - a, d - a])
        volume = np.abs(np.linalg.det(matrix)) / 6
        return volume

      area = tetrahedron_volume(cell_vertices[i])
      self.cell_area[i] = area

  def _set_cell_which_partition(self):
    pass

  def init(self):
    self._set_cell_which_partition()

    ## ghostid
    # self._set_face_nodeid(self.g_cell_nodeid)
    # self._set_g_node_cellid(self.g_cell_nodeid)
    # self._set_ghost_info(self.face_nodeid, self.cell_which_partition) #ghost_info, face_ghostid, node_ghostid
    # self._set_g_cell_ghostnid(self.g_node_ghostid, self.g_cell_nodeid)
    # self._set_l_cell_ghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    # self._set_cell_haloghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)

    # Cell
    self._set_cell_vertices()
    self._set_cell_center(self.cell_vertices)
    self._set_cell_area(self.cell_vertices)
    # self._set_g_cell_cellfid()
    # self._set_g_cell_cellnid()
    # self._set_l_cell_cellfid(self.g_cell_cellfid, self.cell_which_partition)
    # self._set_l_cell_cellnid(self.g_cell_cellnid)
    # self._set_cell_halofid(self.g_cell_cellfid)
    # self._set_cell_halonid(self.g_cell_cellnid)

    ## Face
    # self._set_face_measure(self.cell_vertices)
    # self._set_face_vertices(self.cell_vertices)
    # self._set_face_center(self.faces_vertices)
    # self._set_face_ghostcenter(self.ghost_info, self.face_ghostid)
    # self._set_g_face_cellid()
    # self._set_l_and_g_face_name(self.g_face_cellid, self.cell_which_partition)
    # self._set_l_face_cellid(self.l_face_name, self.g_face_cellid)
    # self._set_face_normal(self.faces_vertices, self.face_center, self.cell_center)
    # self._set_cell_nf(self.face_normal)
    #
    # ## Node
    # self._set_l_node_cellid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    # self._set_node_halonid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    # self._set_node_oldname()
    # self._set_node_name(self.g_node_name, self.l_face_name, self.g_cell_nodeid)
    # self._set_l_node_ghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    # self._set_node_haloghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    #
    # ## Halo
    # self._set_halo_halosint(self.cell_halonid, self.cell_which_partition)
    # self._set_halo_neigh(self.cell_halonid, self.cell_which_partition)
    # self._set_halo_sizehaloghost(self.node_haloghostid, self.cell_which_partition)

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

