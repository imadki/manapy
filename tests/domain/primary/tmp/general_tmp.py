import numpy as np

# TODO check dependencies for general use
# TODO docs for general function
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
    self.nb_faces = -1 #initialize by general
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
    self.cell_halonid = np.ones(shape=(self.nb_cells, 70+1), dtype=np.int32) * -1
    self.cell_halofid = np.ones(shape=(self.nb_cells, 4+1), dtype=np.int32) * -1
    self.g_cell_cellnid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_cell_cellnid = np.ones(shape=(self.nb_cells, 70+1), dtype=np.int32) * -1
    self.g_cell_cellfid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_cell_cellfid = np.ones(shape=(self.nb_cells, 4+1), dtype=np.int32) * -1
    # self.cell_nf = np.zeros(shape=(self.nb_cells, 4, 2), dtype=float_precision)
    self.g_cell_faceid = np.ndarray(shape=(1), dtype=np.int32)  # initialize by general

    self.faces_measure = np.zeros(shape=(self.nb_cells, 4), dtype=float_precision)
    self.g_face_nodeid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.face_center = np.zeros(shape=(self.nb_cells, 4, 3), dtype=float_precision)
    self.face_normal = np.zeros(shape=(self.nb_cells, 4, 3), dtype=float_precision)
    self.faces_vertices = np.zeros(shape=(self.nb_cells, 4, 3, 3), dtype=float_precision)
    # self.face_ghostcenter = np.zeros(shape=(self.nb_cells, 4, 3), dtype=float_precision)
    self.g_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.l_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.g_face_cellid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_face_cellid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1
    # self.face_nodeid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1

    self.g_node_cellid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_node_cellid = np.ones(shape=(self.nb_cells, 4, 25), dtype=np.int32) * -1
    self.node_halonid = np.ones(shape=(self.nb_cells, 4, 25), dtype=np.int32) * -1
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



  def general(self, g_cell_nodeid, nb_nodes, max_face_nodeid, max_cell_faceid, dim):
    """
    create:
      g_node_cellid
      g_cell_cellnid
      g_face_nodeid
      g_cell_faceid
      g_face_cellid
      g_cell_cellfid
      nb_faces
    """

    def _is_in_array(array: 'int[:]', item: 'int') -> 'int':
      """
        Check if an item is in the array
        Return 1 if the item is in the array otherwise 0

        Note:
          The number of item in the array must be array[-1]
      """
      for i in range(array[-1]):
        if item == array[i]:
          return 1
      return 0

    def binary_search(array: 'int[:]', item: 'int') -> 'int':
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


    def count_max_node_cellid(cells: 'int[:, :]', nb_nodes: 'int'):
      """
        Determine the max neighboring cells of a node across all cells
      """
      res = np.zeros(shape=(nb_nodes), dtype=np.int32)
      for cell in cells:
        for i in range(cell[-1]):
          node = cell[i]
          res[node] += 1
      return np.max(res)

    def count_max_cell_cellnid(
            cells: 'int[:, :]',
            node_cellid: 'int[:, :]',
    ):
      """
        Get the maximum number of neighboring cells per cell's nodes across the mesh

        Details:
        For each cell in the mesh, we need to examine its nodes and count the cells that neighbor those nodes.
        to get all neighboring cells of the cell
        Then, determine the highest number of neighboring cells

        Args:
          cells: (cell_id => nodes of the cell)
          node_cellid: (node_id => neighboring cells of the node)

        Return:
          Maximum number of neighboring cells per cell's nodes across the mesh

        Implementation details:
          to ensure that a neighboring cell is visited only once, we set `visited[neighbor_cell] = cell_id`
          thus for the same neighboring cell `visited[neighbor_cell]` is already set by `cell_id`
          for the next cell `visited` will automatically reset because next_cell_id != all_old_cell_id
      """
      visited = np.zeros(cells.shape[0], dtype=np.int32)

      max_counter = 0
      for i in range(cells.shape[0]):
        counter = 0
        for j in range(cells[i][-1]):
          node_n = node_cellid[cells[i][j]]
          for k in range(node_n[-1]):
            if node_n[k] != i and visited[node_n[k]] != i:
              visited[node_n[k]] = i
              counter += 1
        max_counter = max(max_counter, counter)
      return max_counter

    def create_node_cellid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
      """
        Create neighboring cells for each node
      """
      for i in range(cells.shape[0]):
        for j in range(cells[i][-1]):
          node = node_cellid[cells[i][j]]
          size = node[-1]
          node[-1] += 1
          node[size] = i

      for i in range(node_cellid.shape[0]):
        node = node_cellid[i]
        node[0:node[-1]].sort()

    def create_cell_cellnid(
            cells: 'int[:, :]',
            node_cellid: 'int[:, :]',
            cell_cellnid: 'int[:, :]',
    ):
      """
        Get all neighboring cells by collecting adjacent cells from each node of the cell.
      """
      for i in range(cells.shape[0]):
        for j in range(cells[i][-1]):
          node_n = node_cellid[cells[i][j]]
          for k in range(node_n[-1]):
            if node_n[k] != i and _is_in_array(cell_cellnid[i], node_n[k]) == 0:
              size = cell_cellnid[i][-1]
              cell_cellnid[i][-1] += 1
              cell_cellnid[i][size] = node_n[k]

    # ###############
    # Create_info
    # ###############

    def _intersect_nodes(face_nodes: 'int[:]', nb_nodes: 'int', node_cellid: 'int[:, :]',
                         intersect_cell: 'int[:]'):
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
          if binary_search(node_cellid[face_nodes[j]], cells[i]) == -1: #TODO speedup the code by sorting
            intersect_cell[index] = -1
            break
        if intersect_cell[index] != -1:
          index = index + 1
        if index >= 2:
          return

    def _create_cell_faces_n(cells: 'int[:, :]', tmp_cell_faces: 'int[:, :, :]', tmp_size_info: 'int[:, :]', dim: 'int'):
      """
        Create cell faces

        Args:
          nodes : nodes of the cell
          cell_type :
            5 => triangle
            6 => rectangle
            7 => tetrahedron
            9 => hexahedron
            8 => pyramid

        Return:
          out_faces: faces of the cell
          size_info:
            size_info[:-1] contains number of nodes of each face
            size_info[-1] total number of faces of the cell

        Notes:
        'triangle': {'line': [[0, 1], [1, 2], [2, 0]]},
        'rectangle': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]},
        'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
        'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                         [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
        'pyr': {'quad': [[0, 1, 2, 3]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}

      """
      triangle = 5
      rectangle = 6
      tetrahedron = 7
      hexahedron = 8
      pyramid = 9

      for i in range(cells.shape[0]):
        nodes = cells[i]
        out_faces =  tmp_cell_faces[i]
        size_info = tmp_size_info[i]
        cell_type = cells[i][-1] + dim

        if cell_type == triangle:
          out_faces[0][0] = nodes[0]
          out_faces[0][1] = nodes[1]
          size_info[0] = 2  # number of nodes

          out_faces[1][0] = nodes[1]
          out_faces[1][1] = nodes[2]
          size_info[1] = 2

          out_faces[2][0] = nodes[2]
          out_faces[2][1] = nodes[0]
          size_info[2] = 2

          size_info[-1] = 3  # number of faces
        elif cell_type == rectangle:
          out_faces[0][0] = nodes[0]
          out_faces[0][1] = nodes[1]
          size_info[0] = 2  # number of nodes

          out_faces[1][0] = nodes[1]
          out_faces[1][1] = nodes[2]
          size_info[1] = 2

          out_faces[2][0] = nodes[2]
          out_faces[2][1] = nodes[3]
          size_info[2] = 2

          out_faces[3][0] = nodes[3]
          out_faces[3][1] = nodes[0]
          size_info[3] = 2

          size_info[-1] = 4  # number of faces
        elif cell_type == tetrahedron:
          out_faces[0][0] = nodes[0]
          out_faces[0][1] = nodes[1]
          out_faces[0][2] = nodes[2]
          size_info[0] = 3  # number of nodes

          out_faces[1][0] = nodes[0]
          out_faces[1][1] = nodes[1]
          out_faces[1][2] = nodes[3]
          size_info[1] = 3

          out_faces[2][0] = nodes[0]
          out_faces[2][1] = nodes[2]
          out_faces[2][2] = nodes[3]
          size_info[2] = 3

          out_faces[3][0] = nodes[1]
          out_faces[3][1] = nodes[2]
          out_faces[3][2] = nodes[3]
          size_info[3] = 3

          size_info[-1] = 4  # number of faces
        elif cell_type == hexahedron:
          out_faces[0][0] = nodes[0]
          out_faces[0][1] = nodes[1]
          out_faces[0][2] = nodes[2]
          out_faces[0][3] = nodes[3]
          size_info[0] = 4

          out_faces[1][0] = nodes[0]
          out_faces[1][1] = nodes[1]
          out_faces[1][2] = nodes[4]
          out_faces[1][3] = nodes[5]
          size_info[1] = 4

          out_faces[2][0] = nodes[1]
          out_faces[2][1] = nodes[2]
          out_faces[2][2] = nodes[5]
          out_faces[2][3] = nodes[6]
          size_info[2] = 4

          out_faces[3][0] = nodes[2]
          out_faces[3][1] = nodes[3]
          out_faces[3][2] = nodes[6]
          out_faces[3][3] = nodes[7]
          size_info[3] = 4

          out_faces[4][0] = nodes[0]
          out_faces[4][1] = nodes[3]
          out_faces[4][2] = nodes[4]
          out_faces[4][3] = nodes[7]
          size_info[4] = 4

          out_faces[5][0] = nodes[4]
          out_faces[5][1] = nodes[5]
          out_faces[5][2] = nodes[6]
          out_faces[5][3] = nodes[7]
          size_info[5] = 4

          size_info[-1] = 6
        elif cell_type == pyramid:
          out_faces[0][0] = nodes[0]
          out_faces[0][1] = nodes[1]
          out_faces[0][2] = nodes[2]
          out_faces[0][3] = nodes[3]
          size_info[0] = 4

          out_faces[1][0] = nodes[0]
          out_faces[1][1] = nodes[1]
          out_faces[1][2] = nodes[4]
          size_info[1] = 3

          out_faces[2][0] = nodes[1]
          out_faces[2][1] = nodes[2]
          out_faces[2][2] = nodes[4]
          size_info[2] = 3

          out_faces[3][0] = nodes[2]
          out_faces[3][1] = nodes[3]
          out_faces[3][2] = nodes[4]
          size_info[3] = 3

          out_faces[4][0] = nodes[0]
          out_faces[4][1] = nodes[3]
          out_faces[4][2] = nodes[4]
          size_info[4] = 3

          size_info[-1] = 5
        else:
          raise Exception('Unknown cell type')

    def create_info(
      cells: 'int[:, :]',
      node_cellid: 'int[:, :]',
      faces: 'int[:, :]',
      cell_faces: 'int[:, :]',
      face_cellid: 'int[:, :]',
      cell_cellfid: 'int[:, :]',
      faces_counter: 'int[:]',
      tmp_cell_faces: 'int[:, :, :]',
      tmp_size_info: 'int[:, :]',
      tmp_cell_faces_map: 'int[:, :]',
      dim: 'int',
    ):
      """
        - Create faces
        - Create cells with their corresponding faces (cells.cellfid).
        - Create neighboring cells for each face (faces.cellid).
        - Create neighboring cells of a cell by face (cells.cellid).

        Args:
          cells: cells with their nodes (cell => cell nodes)
          node_cellid: neighbor cells of each node (node => neighbor cells)
          max_nb_nodes : maximum number of nodes on faces
          max_nb_faces : maximum number of faces on cells

        Return:
          faces : (face => face nodes)
          cell_faces : (cell => cell faces)
          face_cellid : (face => neighboring cells of the face)
          faces_counter : array(1) face counter
          cell_cellfid : (cell => neighboring cells of a cell by face)

      """

      intersect_cells = np.zeros(2, dtype=np.int32)
      _create_cell_faces_n(cells, tmp_cell_faces, tmp_size_info, dim)
      nb_faces = (tmp_cell_faces_map.shape[1] - 1) // 2

      for i in range(cells.shape[0]):
        # For every face of the cell[i]
        # Get the intersection of the neighboring cells of this face's nodes (N*n*n)
        # The result should be two cells `intersect_cells`
        for j in range(tmp_size_info[i, -1]):
          _intersect_nodes(tmp_cell_faces[i, j], tmp_size_info[i, j], node_cellid, intersect_cells)
          # The face has at most two neighbors
          # swap to make intersect_cells[0] = cell_i id
          if intersect_cells[1] == i:
            intersect_cells[1] = intersect_cells[0]
            intersect_cells[0] = i

          face_id = -1
          # Check if the face already exist
          if intersect_cells[1] != -1:
            for k in range(tmp_cell_faces_map[i, -1]):
              if tmp_cell_faces_map[i, k] == intersect_cells[1]:
                face_id = tmp_cell_faces_map[i, nb_faces + k]

          if face_id == -1:
            face_id = faces_counter[0]
            faces_counter[0] += 1
            # copy nodes from tmp_cell_faces
            for k in range(tmp_size_info[i, j]):
              faces[face_id, k] = tmp_cell_faces[i, j, k]
            faces[face_id, -1] = tmp_size_info[i, j]

            # Store the face in tmp_cell_faces_map for later existence verification.
            if intersect_cells[1] != -1:
              a = tmp_cell_faces_map[intersect_cells[1]]
              size = a[-1]
              a[size] = i
              a[nb_faces + size] = face_id
              a[-1] += 1

          # (cell_faces) Create cell faces
          cell_faces[i, j] = face_id
          cell_faces[i, -1] += 1

          # (face_cellid) Create neighboring cells of each face
          face_cellid[face_id, 0] = intersect_cells[0]
          face_cellid[face_id, 1] = intersect_cells[1]

          # (cell_cellfid) Create neighboring cells of the cell by face
          if intersect_cells[1] != -1:
            cell_cellfid[i, j] = intersect_cells[1]
            cell_cellfid[i, -1] += 1

    # ###############
    # End Create_info
    # ###############

    nb_cells = len(g_cell_nodeid)

    # create_node_cellid
    max_node_cellid = count_max_node_cellid(g_cell_nodeid, nb_nodes)
    node_cellid = np.ones(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32) * -1
    node_cellid[:, -1] = 0
    create_node_cellid(g_cell_nodeid, node_cellid)

    # create_cell_cellnid
    max_cell_cellnid = count_max_cell_cellnid(g_cell_nodeid, node_cellid)
    cell_cellnid = np.ones(shape=(nb_cells, max_cell_cellnid + 1), dtype=np.int32) * -1
    cell_cellnid[:, -1] = 0
    create_cell_cellnid(g_cell_nodeid, node_cellid, cell_cellnid)

    # create_info
    tmp_cell_faces = np.zeros(shape=(nb_cells, max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)


    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.ones(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32) * -1
    faces[:, -1] = 0
    cell_faceid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_faceid[:, -1] = 0
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_cellfid[:, -1] = 0
    face_counter = np.zeros(shape=(1), dtype=np.int32)
    create_info(g_cell_nodeid, node_cellid, faces, cell_faceid, face_cellid, cell_cellfid,
                face_counter, tmp_cell_faces, tmp_size_info, tmp_cell_faces_map, dim)
    faces = faces[:face_counter[0]]
    face_cellid = face_cellid[:face_counter[0]]


    self.g_node_cellid = node_cellid
    self.g_cell_cellnid = cell_cellnid
    self.g_face_nodeid = faces
    self.g_cell_faceid = cell_faceid
    self.g_face_cellid = face_cellid[cell_faceid[:, 0:-1]]
    self.g_cell_cellfid = cell_cellfid
    self.nb_faces = face_counter[0]

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
    for i in range(self.nb_cells):
      center = np.sum(face_vertices[i], axis=1) / 3.0
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

    for i in range(self.nb_cells):
      normal = np.zeros(shape=self.face_normal[0].shape, dtype=self.face_normal.dtype)

      for j in range(len(face_vertices[i])):
        vertices = face_vertices[i, j]
        normal[j] = normal_from_triangle(vertices)
        snorm = tmp_cell_center[i] - face_center[i, j]
        if (np.dot(normal[j], snorm)) > 0:
          normal[j] *= -1

      self.face_normal[i] = normal

  def _set_face_ghostcenter(self, ghost_info, face_ghostid):
    pass

  def _set_face_vertices(self, cell_vertices):
    for i in range(self.nb_cells):
      cellv = cell_vertices[i]
      a = np.array([
        cellv[[0, 1, 2]],
        cellv[[0, 1, 3]],
        cellv[[0, 2, 3]],
        cellv[[1, 2, 3]]
      ])

      self.faces_vertices[i] = a

  def _set_g_face_cellid(self):
    pass #Done by general

  def _set_l_face_cellid(self, l_face_name, g_face_cellid):
    for i in range(self.nb_cells):
      name = l_face_name[i]
      arr = g_face_cellid[i].copy()
      arr[:, 1][name == 10] = -1
      self.l_face_cellid[i] = arr

  def _set_l_and_g_face_name(self, g_face_cellid, face_center, cell_which_partition):
    Width = self.WIDTH
    Height = self.HEIGHT
    Depth = self.DEPTH
    for i in range(self.nb_cells):
      name = np.array([0, 0, 0, 0])
      nb_faces = 4
      for j in range(nb_faces):
        if g_face_cellid[i, j, 1] == -1:
          if face_center[i, j, 0] <= 0.0:
            name[j] = 1
          elif face_center[i, j, 0] >= Width:
            name[j] = 2
          elif face_center[i, j, 1] <= 0.0:
            name[j] = 4
          elif face_center[i, j, 1] >= Height:
            name[j] = 3
          elif face_center[i, j, 2] <= 0.0:
            name[j] = 6
          elif face_center[i, j, 2] >= Depth:
            name[j] = 5

      self.g_face_name[i] = name

      for j in range(nb_faces):
        face_cellid = g_face_cellid[i][j]
        if face_cellid[0] != -1 and face_cellid[1] != -1:
          cell_1_partition = cell_which_partition[face_cellid[0]]
          cell_2_partition = cell_which_partition[face_cellid[1]]
          if cell_1_partition != cell_2_partition:
            name[j] = 10

      self.l_face_name[i] = name

  def _set_cell_nf(self, faces_normal):
    self.cell_nf = faces_normal.copy()


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
    pass #Done by general

  def _set_l_node_cellid(self, g_cell_nodeid, g_node_cellid, which_partition):
    cell_nb_node = 4
    for cell_id in range(self.nb_cells):
      for i in range(cell_nb_node):
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
    cell_nb_node = 4
    for cell_id in range(self.nb_cells):
      for i in range(cell_nb_node):
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
    #Done by general
    for i in range(self.nb_cells):
      arr = self.g_cell_cellfid[i]
      arr = arr[0:-1]
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
    pass #Done by general

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
    nb_partitions = self.nb_partitions
    d_loctoglob = self.d_cell_loctoglob

    for p in range(nb_partitions):
      loctoglob = d_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        self.cell_which_partition[global_index] = p

  def init(self):
    self._set_cell_which_partition()
    # g_node_cellid, g_cell_cellnid, g_cell_faceid, g_face_nodeid, g_face_cellid, g_cell_cellfid, nb_faces
    self.general(self.g_cell_nodeid, self.nb_nodes, 3, 4, 3)

    ## Cell
    self._set_cell_vertices()
    self._set_cell_center(self.cell_vertices)
    self._set_cell_area(self.cell_vertices)
    self._set_g_cell_cellfid()
    self._set_g_cell_cellnid()
    self._set_l_cell_cellfid(self.g_cell_cellfid, self.cell_which_partition)
    self._set_l_cell_cellnid(self.g_cell_cellnid)
    self._set_cell_halofid(self.g_cell_cellfid)
    self._set_cell_halonid(self.g_cell_cellnid)

    ## Face
    self._set_face_measure(self.cell_vertices)
    self._set_face_vertices(self.cell_vertices)
    self._set_face_center(self.faces_vertices)
    self._set_g_face_cellid()
    self._set_l_and_g_face_name(self.g_face_cellid, self.face_center, self.cell_which_partition)
    self._set_l_face_cellid(self.l_face_name, self.g_face_cellid)
    self._set_face_normal(self.faces_vertices, self.face_center, self.cell_center)
    self._set_cell_nf(self.face_normal)

    # ## Node
    self._set_g_node_cellid(self.g_cell_nodeid)
    self._set_l_node_cellid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    self._set_node_halonid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    # self._set_node_oldname()
    # self._set_node_name(self.g_node_name, self.l_face_name, self.g_cell_nodeid)
    #
    # ## ghostid
    # self._set_face_nodeid(self.g_cell_nodeid)
    # self._set_ghost_info(self.face_nodeid, self.cell_which_partition, self.cell_center, self.g_face_name, self.face_center) #ghost_info, face_ghostid, node_ghostid
    # self._set_g_cell_ghostnid(self.g_node_ghostid, self.g_cell_nodeid)
    # self._set_l_cell_ghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    # self._set_cell_haloghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    # self._set_face_ghostcenter(self.ghost_info, self.face_ghostid)
    # self._set_l_node_ghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    # self._set_node_haloghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    #
    # ## Halo
    # self._set_halo_halosint(self.cell_halonid, self.cell_which_partition)
    # self._set_halo_neigh(self.cell_halonid, self.cell_which_partition)
    # self._set_halo_sizehaloghost(self.node_haloghostid, self.cell_which_partition, self.g_cell_nodeid)

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

