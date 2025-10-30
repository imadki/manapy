import numpy as np
import meshio

class TablesTestHybrid:
  def __init__(self, float_precision, d_cell_loctoglob, g_cell_nodeid):
    """
    d_cell_loctoglob: loctoglob of the local domains
    g_cell_nodeid: cell_nodeid of the global domain
    """

    if float_precision != 'float32' and float_precision != 'float64':
      raise ValueError('float_precision must be "float32" or "float64"')

    self.float_precision = float_precision


    self.nb_faces = -1 #initialize by general
    self.nb_cells = self.width * self.width * self.width *  6
    self.nb_nodes = 1331
    self.nb_ghosts = 1200
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
    self.g_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.l_face_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.g_face_cellid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_face_cellid = np.ones(shape=(self.nb_cells, 4, 2), dtype=np.int32) * -1

    self.g_node_cellid = np.ndarray(shape=(1), dtype=np.int32) #initialize by general
    self.l_node_cellid = np.ones(shape=(self.nb_cells, 4, 25), dtype=np.int32) * -1
    self.node_halonid = np.ones(shape=(self.nb_cells, 4, 25), dtype=np.int32) * -1
    self.g_node_name = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.l_node_name = np.ones(shape=self.nb_nodes, dtype=np.int32) * -1

    self.face_nodeid = np.ones(shape=(self.nb_cells, 4, 3), dtype=np.int32) * -1
    self.ghost_info = np.ones(shape=(self.nb_ghosts, 7), dtype=float_precision) * -1
    self.g_cell_ghostnid = np.ones(shape=(self.nb_cells, 20+1), dtype=np.int32) * -1
    self.l_cell_ghostnid = np.ones(shape=(self.nb_cells, 20+1), dtype=np.int32) * -1
    self.cell_haloghostnid = np.ones(shape=(self.nb_cells, 20+1), dtype=np.int32) * -1
    self.face_ghostid = np.ones(shape=(self.nb_cells, 4), dtype=np.int32) * -1
    self.face_ghostcenter = np.zeros(shape=(self.nb_cells, 4, 4), dtype=float_precision)
    self.g_node_ghostid = np.ones(shape=(self.nb_nodes, 7), dtype=np.int32) * -1
    self.l_node_ghostid = np.ones(shape=(self.nb_cells, 4, 7), dtype=np.int32) * -1
    self.node_haloghostid = np.ones(shape=(self.nb_cells, 4, 7), dtype=np.int32) * -1

    self.halo_halosint = np.array([], np.int32)
    self.halo_neigh = np.zeros(shape=(self.nb_partitions, self.nb_partitions), dtype=np.int32)
    self.halo_sizehaloghost = np.zeros(shape=(self.nb_partitions), dtype=np.int32)



  def general(self, g_cell_nodeid, nb_nodes, max_face_nodeid, max_cell_faceid, dim):
    """
    create:
      g_node_cellid
      g_cell_faceid
      g_cell_cellfid
      g_cell_cellnid
      g_face_nodeid
      g_face_cellid
      nb_faces
    """

    def _is_in_array(array: 'int[:]', item: 'int') -> 'int':
      """
        Check if an item is inside array with size array[-1]
        Return index >= 0 if the item is in the array otherwise -1
      """
      for i in range(array[-1]):
        if item == array[i]:
          return i
      return -1

    def binary_search(array: 'int[:]', item: 'int') -> 'int':
      """
        Check if an item is inside the sorted array with size array[-1]
        Return index >= 0 if the item is in the array otherwise -1
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

      # sort to use binary search latter
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
            if node_n[k] != i and _is_in_array(cell_cellnid[i], node_n[k]) == -1:
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
          if binary_search(node_cellid[face_nodes[j]], cells[i]) == -1: #? node_cellid must be sorted
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
          cells : [cell => nodes]
          dim: mesh dimension
          cell_type :
            5 => triangle
            6 => rectangle
            7 => tetrahedron
            9 => hexahedron
            8 => pyramid

        Return:
          tmp_cell_faces: array of shape [nb_cells, max_cell_nb_faces, max_face_nb_nodes]
            to store cell face's nodes as described in the Notes
          tmp_size_info:
            tmp_size_info[:, :-1] contains number of nodes of each face
            tmp_size_info[:, -1] total number of faces of the cell

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
          tmp_cell_faces : temporary table to store face's nodes
          tmp_size_info : temporary table to store number of faces's nodes and cell's faces
          tmp_cell_faces_map : temporary table to store a map from face second neighbor cell and the face_id
          dim : dimension of the mesh

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
    self.max_cell_faceid = max_cell_faceid

  def _set_face_measure(self, cells, nodes):

    def _triangle_area_3d(points):
      a = points[0]
      b = points[1]
      c = points[2]

      u = b - a
      v = c - a

      # cross
      cross_x = u[1] * v[2] - u[2] * v[1]
      cross_y = u[2] * v[0] - u[0] * v[2]
      cross_z = u[0] * v[1] - u[1] * v[0]
      area = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
      return area * 0.5

    def _rectangle_area_3d(points):
      return _triangle_area_3d(points[[0, 1, 2]]) + _triangle_area_3d(points[[0, 3, 2]])

    def _measure_2d(points):
      u = points[0] - points[1]
      return np.sqrt(u[0] * u[0] + u[1] * u[1])

    self.faces_measure = np.zeros(shape=(self.nb_cells, self.max_cell_faceid), dtype=self.float_precision)

    # 2D
    if self.dim == 2:
      for i in range(self.nb_cells):
        if cells[i, -1] == 3: #triangle
          self.faces_measure[i] = np.array([
            _measure_2d(nodes[cells[i, [0, 1]]]),
            _measure_2d(nodes[cells[i, [1, 2]]]),
            _measure_2d(nodes[cells[i, [2, 0]]]),
          ])
        elif cells[i, -1] == 4: #quad
          self.faces_measure[i] = np.array([
            _measure_2d(nodes[cells[i, [0, 1]]]),
            _measure_2d(nodes[cells[i, [1, 2]]]),
            _measure_2d(nodes[cells[i, [2, 3]]]),
            _measure_2d(nodes[cells[i, [3, 0]]]),
          ])

    # 3D
    if self.dim == 3:
      for i in range(self.nb_cells):
        if cells[i, -1] == 5: #pyramid
          self.faces_measure[i] = np.array([
            _rectangle_area_3d(nodes[cells[i, [0, 1, 2, 3]]]),
            _triangle_area_3d(nodes[cells[i, [0, 1, 4]]]),
            _triangle_area_3d(nodes[cells[i, [1, 2, 4]]]),
            _triangle_area_3d(nodes[cells[i, [2, 3, 4]]]),
            _triangle_area_3d(nodes[cells[i, [0, 3, 4]]]),
          ])
        elif cells[i, -1] == 8: #hex
          self.faces_measure[i] = np.array([
            _rectangle_area_3d(nodes[cells[i, [0, 1, 2, 3]]]),
            _rectangle_area_3d(nodes[cells[i, [0, 1, 4, 5]]]),
            _rectangle_area_3d(nodes[cells[i, [1, 2, 5, 6]]]),
            _rectangle_area_3d(nodes[cells[i, [2, 3, 6, 7]]]),
            _rectangle_area_3d(nodes[cells[i, [0, 3, 4, 7]]]),
            _rectangle_area_3d(nodes[cells[i, [4, 5, 6, 7]]]),
          ])
        elif cells[i, -1] == 4: #tetra
          self.faces_measure[i] = np.array([
            _triangle_area_3d(nodes[cells[i, [0, 1, 2]]]),
            _triangle_area_3d(nodes[cells[i, [0, 1, 3]]]),
            _triangle_area_3d(nodes[cells[i, [0, 2, 3]]]),
            _triangle_area_3d(nodes[cells[i, [1, 2, 3]]]),
          ])

  def _set_face_center(self, cells, nodes):
    self.faces_measure = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, self.dim), dtype=self.float_precision)

    def _get_center(points):
      return np.sum(points, axis=0) / len(points)

    # 2D
    if self.dim == 2:
      for i in range(self.nb_cells):
        if cells[i, -1] == 3:  # triangle
          self.faces_measure[i] = np.array([
            _get_center(nodes[cells[i, [0, 1]]]),
            _get_center(nodes[cells[i, [1, 2]]]),
            _get_center(nodes[cells[i, [2, 0]]]),
          ])
        elif cells[i, -1] == 4:  # quad
          self.faces_measure[i] = np.array([
            _get_center(nodes[cells[i, [0, 1]]]),
            _get_center(nodes[cells[i, [1, 2]]]),
            _get_center(nodes[cells[i, [2, 3]]]),
            _get_center(nodes[cells[i, [3, 0]]]),
          ])

    # 3D
    if self.dim == 3:
      for i in range(self.nb_cells):
        if cells[i, -1] == 5:  # pyramid
          self.faces_measure[i] = np.array([
            _get_center(nodes[cells[i, [0, 1, 2, 3]]]),
            _get_center(nodes[cells[i, [0, 1, 4]]]),
            _get_center(nodes[cells[i, [1, 2, 4]]]),
            _get_center(nodes[cells[i, [2, 3, 4]]]),
            _get_center(nodes[cells[i, [0, 3, 4]]]),
          ])
        elif cells[i, -1] == 8:  # hex
          self.faces_measure[i] = np.array([
            _get_center(nodes[cells[i, [0, 1, 2, 3]]]),
            _get_center(nodes[cells[i, [0, 1, 4, 5]]]),
            _get_center(nodes[cells[i, [1, 2, 5, 6]]]),
            _get_center(nodes[cells[i, [2, 3, 6, 7]]]),
            _get_center(nodes[cells[i, [0, 3, 4, 7]]]),
            _get_center(nodes[cells[i, [4, 5, 6, 7]]]),
          ])
        elif cells[i, -1] == 4:  # tetra
          self.faces_measure[i] = np.array([
            _get_center(nodes[cells[i, [0, 1, 2]]]),
            _get_center(nodes[cells[i, [0, 1, 3]]]),
            _get_center(nodes[cells[i, [0, 2, 3]]]),
            _get_center(nodes[cells[i, [1, 2, 3]]]),
          ])

  def _set_face_normal(self, cells, nodes):
    def _triangle_normal_3d(points):
      a = points[0]
      b = points[1]
      c = points[2]
      u = b - a
      v = c - a

      # cross
      cross = np.zeros(shape=3, dtype=a.dtype)
      cross[0] = u[1] * v[2] - u[2] * v[1]
      cross[1] = u[2] * v[0] - u[0] * v[2]
      cross[2] = u[0] * v[1] - u[1] * v[0]
      return np.abs(cross)

    def _normal_2d(points):
      return np.abs(np.array([-points[1], points[0]]))

    self.faces_normal = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, self.dim), dtype=self.float_precision)

    # 2D
    if self.dim == 2:
      for i in range(self.nb_cells):
        if cells[i, -1] == 3: #triangle
          self.faces_normal[i] = np.array([
            _normal_2d(nodes[cells[i, [0, 1]]]),
            _normal_2d(nodes[cells[i, [1, 2]]]),
            _normal_2d(nodes[cells[i, [2, 0]]]),
          ])
        elif cells[i, -1] == 4: #quad
          self.faces_normal[i] = np.array([
            _normal_2d(nodes[cells[i, [0, 1]]]),
            _normal_2d(nodes[cells[i, [1, 2]]]),
            _normal_2d(nodes[cells[i, [2, 3]]]),
            _normal_2d(nodes[cells[i, [3, 0]]]),
          ])

    # 3D
    if self.dim == 3:
      for i in range(self.nb_cells):
        if cells[i, -1] == 5: #pyramid
          self.faces_normal[i] = np.array([
            _triangle_normal_3d(nodes[cells[i, [0, 1, 2, 3]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [0, 1, 4]]]),
            _triangle_normal_3d(nodes[cells[i, [1, 2, 4]]]),
            _triangle_normal_3d(nodes[cells[i, [2, 3, 4]]]),
            _triangle_normal_3d(nodes[cells[i, [0, 3, 4]]]),
          ])
        elif cells[i, -1] == 8: #hex
          self.faces_normal[i] = np.array([
            _triangle_normal_3d(nodes[cells[i, [0, 1, 2, 3]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [0, 1, 4, 5]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [1, 2, 5, 6]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [2, 3, 6, 7]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [0, 3, 4, 7]]]) * 2,
            _triangle_normal_3d(nodes[cells[i, [4, 5, 6, 7]]]) * 2,
          ])
        elif cells[i, -1] == 4: #tetra
          self.faces_normal[i] = np.array([
            _triangle_normal_3d(nodes[cells[i, [0, 1, 2]]]),
            _triangle_normal_3d(nodes[cells[i, [0, 1, 3]]]),
            _triangle_normal_3d(nodes[cells[i, [0, 2, 3]]]),
            _triangle_normal_3d(nodes[cells[i, [1, 2, 3]]]),
          ])


  def _set_l_face_cellid(self, l_face_name, cell_faceid, face_cellid):
    self.l_face_cellid = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 2), dtype=np.int32)

    for i in range(self.nb_cells):
      nb_faces = cell_faceid[i, -1]
      for j in range(nb_faces):
        name = l_face_name[i, j]
        n_cells = face_cellid[cell_faceid[i, j]]
        if n_cells[0] != i and name == 10:
          n_cells[0] = -1
        elif n_cells[1] != i and name == 10:
          n_cells[1] = -1
        self.l_face_cellid[i, j] = n_cells

  def _set_l_and_g_face_name(self, phy_faces, faces, cell_faceid, face_cellid, cell_part, node_phyid, phy_faces_name):
    def _get_phyid(
            phy_faces: 'int32[:, :]',
            face_nodes: 'int32[:]',
            node_phyid: 'int32[:, :]',
    ):
      sorted_face_node = np.sort(face_nodes[0:face_nodes[-1]])
      n = face_nodes[0]  # select any node, choosing node 0
      for k in range(node_phyid[n, -1]):
        phyid = node_phyid[n, k]
        phyid_nodes = phy_faces[phyid][0:-1]
        phyid_nodes.sort()
        if np.all(phyid_nodes == sorted_face_node):
          return phyid
      return -1

    self.g_face_name = np.zeros(shape=(self.nb_cells, self.max_cell_faceid), dtype=np.int32)
    self.l_face_name = np.zeros(shape=(self.nb_cells, self.max_cell_faceid), dtype=np.int32)

    for i in range(self.nb_cells):
      nb_faces = cell_faceid[i, -1]
      for j in range(nb_faces):
        face = faces[cell_faceid[i, j]]
        phyid = _get_phyid(phy_faces, face, node_phyid)
        name = 0 if phyid == -1 else phy_faces_name[phyid]


        self.g_face_name[i, j] = name
        self.l_face_name[i, j] = name
        f_cell1 = face_cellid[face, 0]
        f_cell2 = face_cellid[face, 1]
        if f_cell2 != -1 and cell_part[f_cell1] != cell_part[f_cell2]:
          self.l_face_name[i, j] = 10


  def _set_cell_nf(self, faces_normal):
    self.cell_nf = faces_normal.copy()


  ###################
  ## Ghost Info
  ###################

  def _set_ghost_info(self, cells, cell_faceid, faces, face_nodeid, cell_which_partition, cell_center, g_face_name, face_center, face_normal):
    pass
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
      arr[6] = faceid  # faceid in the cell (0..4)
      self.ghost_info[ghostid][:] = arr
      # Set face ghost_id
      self.face_ghostid[cellid, faceid] = ghostid
      face_id = cell_faceid[cellid, faceid]
      nb_face_nodes = faces[face_id, -1]
      for i in range(nb_face_nodes): #each face has 3 nodes
        add_node_ghostid(self.g_node_ghostid[faces[face_id, -1]], ghostid=ghostid)

    cmp = 0
    for i in range(self.nb_cells):
      c_center = cell_center[i]
      nb_faces = cells[i, -1]
      for j in range(nb_faces): #each cell has 4 faces
        f_center = face_center[i, j]
        f_oldname = g_face_name[i, j]
        if f_oldname != 0:
          N = face_normal[i, j]
          n_hat = N / np.linalg.norm(N)
          ghostcenter = c_center - 2 * np.dot(c_center - f_center, n_hat) * n_hat
          add_ghost(ghostcenter, i, cmp, j)
          cmp += 1



  def _set_face_ghostcenter(self, cell_faceid, ghost_info, face_ghostid):
    # self.face_ghostcenter => [center_x center_y center_z gamma]
    for i in range(self.nb_cells):
      for j in range(cell_faceid[i, -1]):
        if face_ghostid[i, j] != -1:
          self.face_ghostcenter[i][j][:] = ghost_info[face_ghostid[i, j]][0:4]
        else:
          self.face_ghostcenter[i][j][:] = -1

  def _set_l_node_ghostid(self, cells, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      nb_nodes = cells[cell_id, -1]
      for i in range(0, nb_nodes): # number of nodes
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        ghost_partition_id = ghost_info[arr][:, 4]
        not_the_same_partition = (ghost_partition_id != cell_partition_id)
        arr[not_the_same_partition] = -1
        arr = arr[arr != -1]
        self.l_node_ghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.l_node_ghostid[cell_id, i, -1] = len(arr)

  def _set_node_haloghostid(self, cells, ghost_info, g_node_ghostid, g_cell_nodeid, cell_which_partition):
    for cell_id in range(self.nb_cells):
      cell_partition_id = cell_which_partition[cell_id]
      cell_nodes = g_cell_nodeid[cell_id][0:g_cell_nodeid[cell_id][-1]]
      nb_nodes = cells[cell_id, -1]
      for i in range(0, nb_nodes):
        arr = g_node_ghostid[cell_nodes[i]].copy()
        arr = arr[0:arr[-1]]
        ghost_partition_id = ghost_info[arr][:, 4]
        same_partition = (ghost_partition_id == cell_partition_id)
        arr[same_partition] = -1
        arr = arr[arr != -1]
        self.node_haloghostid[cell_id, i, 0:len(arr)] = arr[:]
        self.node_haloghostid[cell_id, i, -1] = len(arr)

  def _set_g_cell_ghostnid(self, cells, g_node_ghostid, g_cell_nodeid):
    for cell_id in range(self.nb_cells):
      nb_nodes = cells[cell_id, -1]
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

  def _set_l_node_cellid(self, g_cell_nodeid, g_node_cellid, which_partition):
    self.node_halonid = np.zeros(shape=(self.nb_cells, self.max_cell_nodeid, self.max_node_cellid + 1), dtype=np.int32)
    for cell_id in range(self.nb_cells):
      nb_nodes = g_cell_nodeid[cell_id, -1]
      for i in range(nb_nodes):
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
    self.node_halonid = np.zeros(shape=(self.nb_cells, self.max_cell_nodeid, self.max_node_haloid + 1), dtype=np.int32)
    # The same as set_l_node_cellid
    for cell_id in range(self.nb_cells):
      nb_nodes = g_cell_nodeid[cell_id, -1]
      for i in range(nb_nodes):
        node_id = g_cell_nodeid[cell_id, i]
        node_cellid = g_node_cellid[node_id]
        node_cellid = node_cellid[0:node_cellid[-1]]
        node_cellid = node_cellid.copy()
        this_cell_partition = which_partition[cell_id]
        node_cellid[which_partition[node_cellid] == this_cell_partition] = -1 #equal instead of !=
        node_cellid = node_cellid[node_cellid != -1]

        self.node_halonid[cell_id][i][0:len(node_cellid)] = node_cellid[:]
        self.node_halonid[cell_id][i][-1] = len(node_cellid)

  def _set_node_oldname(self, cells, node_phyid, phy_faces_name):
    self.g_node_name = np.zeros(shape=(self.nb_cells, self.max_cell_nodeid), dtype=np.int32)

    for i in range(self.nb_cells):
      nb_nodes = cells[i, -1]
      for j in range(nb_nodes):
        node = cells[i, j]
        phys = node_phyid[node, 0:node_phyid[node, -1]]
        names = phy_faces_name[phys]
        self.g_node_name[i, j] = 0
        if len(names) != 0:
          self.g_node_name[i, j] = min(names)

  def _set_node_name(self, cells, g_node_name, node_halonid):
    self.l_node_name = np.zeros(shape=(self.nb_cells, self.max_cell_nodeid), dtype=np.int32)

    for i in range(self.nb_cells):
      nb_nodes = cells[i, -1]
      for j in range(nb_nodes):
        self.l_node_name[i, j] = g_node_name[i, j]
        nb_halos = node_halonid[i, j, -1]
        if nb_halos != 0:
          self.g_node_name[i, j] = 10


  ###################
  ## Halo Info
  ###################

  def _set_halo_halosint(self, cell_halonid, cell_parts):
    res = [[] for i in range(self.nb_partitions)]
    for i in range(len(cell_halonid)):
      item = cell_halonid[i]
      p = cell_parts[i]
      if np.any(item != -1):
        res[p].append(i)

    max_len = max(len(subarray) for subarray in res)
    res = [subarray + [-1] * (max_len - len(subarray)) for subarray in res]
    self.halo_halosint = np.array(res, dtype=np.int32)

  def _set_halo_neigh(self, cell_halonid, cell_parts):
    haloext = [[] for i in range(self.nb_partitions)]
    for i in range(len(cell_halonid)):
      item = cell_halonid[i]
      p = cell_parts[i]
      haloext[p] += list(item[item != -1])

    for p in range(self.nb_partitions):
      for j in range(self.nb_partitions):
        tmp = np.array(haloext[p], dtype=np.int32)
        tmp = np.unique(tmp)
        self.halo_neigh[p][j] = np.sum(cell_parts[tmp] == j) #ext that belong to partition j


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


  def _set_l_cell_cellfid(self, g_cell_cellfid, cell_which_partition):
    for i in range(self.nb_cells):
      arr = (g_cell_cellfid[i][0:g_cell_cellfid[i][-1]]).copy()
      this_cell_partition = cell_which_partition[i]
      arr[cell_which_partition[arr] != this_cell_partition] = -1
      arr = arr[arr != -1]

      self.l_cell_cellfid[i][0:arr.shape[0]] = arr
      self.l_cell_cellfid[i][-1] = arr.shape[0]


  def _set_l_cell_cellnid(self, g_cell_cellnid):
    for i in range(self.nb_cells):
      arr = (g_cell_cellnid[i][0:g_cell_cellnid[i][-1]]).copy()
      this_cell_partition = self.cell_which_partition[i]
      arr[self.cell_which_partition[arr] != this_cell_partition] = -1
      arr = arr[arr != -1]

      self.l_cell_cellnid[i][0:arr.shape[0]] = arr
      self.l_cell_cellnid[i][-1] = arr.shape[0]

  def _read_mesh(self, mesh_path, dim):
    # TODO max_cell_faces

    # Read Mesh
    mesh = meshio.read(mesh_path)
    MESHIO_VERSION = int(meshio.__version__.split(".")[0])
    if MESHIO_VERSION < 4:
      cells_dict = mesh.cells
    else:
      cells_dict = mesh.cells_dict

    # Construct Points
    points = mesh.points
    points = np.array(points, dtype=self.float_precision)

    # Construct Cells
    allowed_cells = ['quad', 'triangle']
    if dim == 3:
      allowed_cells = ['pyramid', 'hexahedron', 'tetra']

    max_cell_nodeid = -1
    for item in cells_dict.keys():
      if item == 'triangle':
        max_cell_nodeid = max(max_cell_nodeid, 3)
      elif item == 'quad':
        max_cell_nodeid = max(max_cell_nodeid, 4)
      elif item == 'tetra':
        max_cell_nodeid = max(max_cell_nodeid, 4)
      elif item == 'hexahedron':
        max_cell_nodeid = max(max_cell_nodeid, 8)
      elif item == 'pyramid':
        max_cell_nodeid = max(max_cell_nodeid, 5)

    number_of_cells = 0
    for item in allowed_cells:
      if cells_dict.get(item) is not None:
        number_of_cells += len(cells_dict[item])

    cells = np.zeros(shape=(number_of_cells, max_cell_nodeid + 1), dtype=np.int32)

    counter = np.int32(0)
    for item in allowed_cells:
      if cells_dict.get(item) is not None:
        cells_item = np.array(cells_dict[item], dtype=np.int32)
        for i in range(len(cells_item)):
          cells[counter, 0:len(cells_item[i])] = cells_item[i]
          cells[counter, -1] = len(cells_item[i])
          counter += 1

    self.cells = cells
    self.nodes = points
    self.dim = dim
    #self.max_cell_nodeid = max_cell_nodeid
    self.nb_cells = number_of_cells

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

  def _set_cell_center(self, cells, nodes):
    self.cell_center = np.zeros(shape=(self.nb_cells, self.dim), dtype=self.float_precision)
    for i in range(self.nb_cells):
      points = nodes[cells[i, 0:cells[i, -1]]]
      self.cell_center[i, :] = np.sum(points, axis=0) / len(points)

  def _set_cell_area(self, cells, nodes):

    def _polygon_area_2d(points: 'float[:, :]'):
      n = len(points)
      area = 0.0
      for i in range(n):
        x1 = points[i, 0]
        y1 = points[i, 1]
        x2 = points[(i + 1) % n, 0]
        y2 = points[(i + 1) % n, 1]
        area += (x1 * y2) - (x2 * y1)
      return abs(area) / 2.0

    def _tetrahedron_volume(a: 'float[:]', b: 'float[:]', c: 'float[:]', d: 'float[:]'):
      # Compute det of [b - a, c - a, d - a] matrix
      u = b - a
      v = c - a
      w = d - a

      cross_x = v[1] * w[2] - v[2] * w[1]
      cross_y = v[2] * w[0] - v[0] * w[2]
      cross_z = v[0] * w[1] - v[1] * w[0]

      det = (u[0] * cross_x + u[1] * cross_y + u[2] * cross_z)
      volume = det / 6
      return volume

    self.cell_area = np.zeros(shape=self.nb_cells, dtype=self.float_precision)
    if self.dim == 3:
      for i in range(len(cells)):
        nb_vertex = cells[i, -1]
        points = nodes[cells[i, 0:nb_vertex]]

        # Volume
        vol = 0.0
        if nb_vertex == 4:  # Tetrahedron
          vol += _tetrahedron_volume(points[0], points[1], points[2], points[3])
        elif nb_vertex == 8:  # Hexahedron
          # [0, 1, 3, 4], # 1 tetra
          # [1, 3, 4, 5], # 2 tetra
          # [4, 5, 3, 7], # 3 tetra
          # [1, 3, 5, 2], # 4 tetra
          # [3, 7, 5, 2], # 5 tetra
          # [5, 7, 6, 2]  # 6 tetra
          vol += _tetrahedron_volume(points[0], points[1], points[3], points[4])
          vol += _tetrahedron_volume(points[1], points[3], points[4], points[5])
          vol += _tetrahedron_volume(points[4], points[5], points[3], points[7])
          vol += _tetrahedron_volume(points[1], points[3], points[5], points[2])
          vol += _tetrahedron_volume(points[3], points[7], points[5], points[2])
          vol += _tetrahedron_volume(points[5], points[7], points[6], points[2])
        elif nb_vertex == 5:  # Pyramid
          # [0, 1, 2, 4],  # 1 tetra
          # [0, 2, 3, 4],  # 2 tetra
          vol += _tetrahedron_volume(points[0], points[1], points[2], points[4])
          vol += _tetrahedron_volume(points[0], points[2], points[3], points[4])
        self.cell_area[i] = vol
    else:
      for i in range(len(cells)):
        nb_vertex = cells[i, -1]
        vertices = nodes[cells[i, 0:nb_vertex]]
        self.cell_area[i] = _polygon_area_2d(vertices)

  def _set_cell_which_partition(self):
    nb_partitions = self.nb_partitions
    d_loctoglob = self.d_cell_loctoglob

    for p in range(nb_partitions):
      loctoglob = d_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        self.cell_which_partition[global_index] = p

  def init(self, dim, mesh_path):
    self._set_cell_which_partition()
    # g_node_cellid, g_cell_cellnid, g_cell_faceid, g_face_nodeid, g_face_cellid, g_cell_cellfid, nb_faces
    self.general(self.g_cell_nodeid, self.nb_nodes, 3, 4, 3)

    ## Cell
    self._read_mesh(mesh_path, dim) # cells, points, dim, nb_cells
    self._set_cell_center(self.cells, self.nodes)
    self._set_cell_area(self.cells, self.nodes)
    # self._set_g_cell_cellfid() # done by general
    # self._set_g_cell_cellnid() # done by general
    self._set_l_cell_cellfid(self.g_cell_cellfid, self.cell_which_partition)
    self._set_l_cell_cellnid(self.g_cell_cellnid)
    self._set_cell_halofid(self.g_cell_cellfid)
    self._set_cell_halonid(self.g_cell_cellnid)

    ## Face
    self._set_face_measure(self.cells, self.nodes)
    self._set_face_center(self.cells, self.nodes)
    # self._set_g_face_cellid() # done by general
    self._set_l_and_g_face_name(self.g_face_cellid, self.face_center, self.cell_which_partition)
    # self._set_l_face_cellid(self.l_face_name, self.g_face_cellid)
    # self._set_face_normal(self.faces_vertices, self.face_center, self.cell_center)
    # self._set_cell_nf(self.face_normal)
    #
    # # ## Node
    # self._set_l_node_cellid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    # self._set_node_halonid(self.g_cell_nodeid, self.g_node_cellid, self.cell_which_partition)
    # self._set_node_oldname(self.cell_vertices)
    # self._set_node_name(self.g_node_name, self.l_face_name, self.g_cell_nodeid)
    #
    # # ## ghostid
    self._set_ghost_info(self.face_nodeid, self.cell_which_partition, self.cell_center, self.g_face_name, self.face_center, self.face_normal) #ghost_info, face_ghostid, node_ghostid
    self._set_g_cell_ghostnid(self.g_node_ghostid, self.g_cell_nodeid)
    self._set_l_cell_ghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_cell_haloghostnid(self.g_cell_ghostnid, self.ghost_info, self.cell_which_partition)
    self._set_face_ghostcenter(self.ghost_info, self.face_ghostid)
    self._set_l_node_ghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)
    self._set_node_haloghostid(self.ghost_info, self.g_node_ghostid, self.g_cell_nodeid, self.cell_which_partition)

    # # ## Halo
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

