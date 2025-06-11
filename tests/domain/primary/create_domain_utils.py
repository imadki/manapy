import numpy as np
import numba





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

def _append(cells: 'int[:, :]', cells_item: 'int[:, :]', counter: 'int'):
  for i in range(len(cells_item)):
    cells[counter, 0:len(cells_item[i])] = cells_item[i]
    cells[counter, -1] = len(cells_item[i])
    counter += 1

def _append_1d(arr_dest: 'int[:]', arr_src: 'int[:]', counter):
  for i in range(len(arr_src)):
    arr_dest[counter] = arr_src[i]
    counter += 1

def _count_max_node_cellid(cells: 'int[:, :]', res: 'int[:]'):
  """
    Determine the max neighboring cells of a node across all cells
  """

  for cell in cells:
    for i in range(cell[-1]):
      node = cell[i]
      res[node] += 1


def _create_node_cellid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
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


def _count_max_cell_cellnid(cells: 'int[:, :]', node_cellid: 'int[:, :]', visited: 'int[:]'):
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


def _create_cell_cellnid(
  cells: 'int[:, :]',
  node_cellid: 'int[:, :]',
  cell_cellnid: 'int[:, :]'
):
  """
    Get all neighboring cells by collecting adjacent cells from each node of the cell.
  """

  for i in range(cells.shape[0]):
    for j in range(cells[i, -1]):
      node_n = node_cellid[cells[i][j]]
      for k in range(node_n[-1]):
        nc = node_n[k]
        size = cell_cellnid[nc, -1]
        if nc != i and (size == 0 or cell_cellnid[nc, size - 1] != i):
          cell_cellnid[nc, size] = i
          cell_cellnid[nc, -1] += 1

  # for i in range(cells.shape[0]):
  #   for j in range(cells[i][-1]):
  #     node_n = node_cellid[cells[i][j]]
  #     for k in range(node_n[-1]):
  #       if node_n[k] != i and _is_in_array(cell_cellnid[i], node_n[k]) == 0:
  #         size = cell_cellnid[i][-1]
  #         cell_cellnid[i][-1] += 1
  #         cell_cellnid[i][size] = node_n[k]

# #################
# Create Info
# #################


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
      if _binary_search(node_cellid[face_nodes[j]], cells[i]) == -1:
        intersect_cell[index] = -1
        break
    if intersect_cell[index] != -1:
      index = index + 1
    if index >= 2:
      return

def _create_cell_faces(nodes: 'int[:]', out_faces: 'int[:, :]', size_info: 'int[:]', cell_type: 'int[:]'):
  """
    Create cell faces

    Args:
      nodes : nodes of the cell
      cell_type :
        1 => triangle
        2 => rectangle
        3 => tetrahedron
        4 => hexahedron
        5 => pyramid

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
  triangle = 1
  rectangle = 2
  tetrahedron = 3
  hexahedron = 4
  pyramid = 5


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

def _polygon_area_2d(points : 'float[:, :]'):
  n = len(points)
  area = 0.0
  for i in range(n):
    x1 = points[i, 0]
    y1 = points[i, 1]
    x2 = points[(i + 1) % n, 0]
    y2 = points[(i + 1) % n, 1]
    area += (x1 * y2) - (x2 * y1)
  return abs(area) / 2.0

def _compute_cell_center_volume_2d(cells: 'int[:, :]', nodes: 'float[:, :]', cell_area: 'float[:]', cell_center: 'float[:, :]'):
  for i in range(len(cells)):
    nb_vertex = cells[i, -1]
    vertices = nodes[cells[i, 0:nb_vertex]]
    cell_area[i] = _polygon_area_2d(vertices)
    center = np.sum(vertices, axis=0) / nb_vertex
    cell_center[i] = center[0:2]

def _tetrahedron_volume(a : 'float[:]', b : 'float[:]', c: 'float[:]', d : 'float[:]'):
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

def _compute_cell_center_volume_3d(cells: 'int[:, :]', nodes: 'float[:, :]', cell_area: 'float[:]', cell_center: 'float[:, :]'):
  for i in range(len(cells)):
    nb_vertex = cells[i, -1]
    points = nodes[cells[i, 0:nb_vertex]]

    # Center
    center = np.sum(points, axis=0) / nb_vertex
    cell_center[i] = center[0:3]

    # Volume
    vol = 0.0
    if nb_vertex == 4: # Tetrahedron
      vol += _tetrahedron_volume(points[0], points[1], points[2], points[3])
    elif nb_vertex == 8: # Hexahedron
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
    elif nb_vertex == 5: # Pyramid
      # [0, 1, 2, 4],  # 1 tetra
      # [0, 2, 3, 4],  # 2 tetra
      vol += _tetrahedron_volume(points[0], points[1], points[2], points[4])
      vol += _tetrahedron_volume(points[0], points[2], points[3], points[4])
    cell_area[i] = vol


def _triangle_area_3d(a: 'float[:]', b: 'float[:]', c: 'float[:]'):
  u = b - a
  v = c - a

  # cross
  cross_x = u[1] * v[2] - u[2] * v[1]
  cross_y = u[2] * v[0] - u[0] * v[2]
  cross_z = u[0] * v[1] - u[1] * v[0]
  area = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
  return area * 0.5


def _triangle_normal_3d(a: 'float[:]', b: 'float[:]', c: 'float[:]'):
  u = b - a
  v = c - a

  # cross
  cross = np.zeros(shape=3, dtype=a.dtype)
  cross[0] = u[1] * v[2] - u[2] * v[1]
  cross[1] = u[2] * v[0] - u[0] * v[2]
  cross[2] = u[0] * v[1] - u[1] * v[0]
  return cross


def _compute_face_info_2d(faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]', face_measure : 'float[:]', face_center: 'float[:, :]', face_normal: 'float[:, :]'):
  for i in range(len(faces)):
    nb_vertex = faces[i, -1]
    points = nodes[faces[i, 0:nb_vertex]]

    # Face Measure
    u = points[0] - points[1]
    measure = np.sqrt(u[0] * u[0] + u[1] * u[1])
    face_measure[i] = measure

    # Center
    center = np.sum(points, axis=0) / nb_vertex
    face_center[i] = center[0:2]

    # Face Normal
    normal = np.array([-u[1], u[0]], dtype=u.dtype)
    snorm = cell_center[face_cellid[i, 0]] - center
    if (np.dot(normal, snorm)) > 0:
      normal *= -1
    face_normal[i] = normal

def _compute_face_info_3d(faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]', face_measure : 'float[:]', face_center: 'float[:, :]', face_normal: 'float[:, :]'):
  for i in range(len(faces)):
    nb_vertex = faces[i, -1]
    points = nodes[faces[i, 0:nb_vertex]]

    measure = 0
    normal = np.zeros(shape=3, dtype=points.dtype)
    if nb_vertex == 3: #Triangle
      measure = _triangle_area_3d(points[0], points[1], points[2])
      normal[:] = _triangle_normal_3d(points[0], points[1], points[2])
    elif nb_vertex == 4: #Rectangle
      measure = _triangle_area_3d(points[0], points[1], points[2]) * 2
      normal[:] = _triangle_normal_3d(points[0], points[1], points[2]) * 2

    # Face Measure
    face_measure[i] = measure

    # Center
    center = np.sum(points, axis=0) / nb_vertex
    face_center[i] = center[0:3]

    # Face Normal
    snorm = cell_center[face_cellid[i, 0]] - center
    if (np.dot(normal, snorm)) > 0:
      normal *= -1
    face_normal[i] = normal



def _create_info(
  cells: 'int[:, :]',
  node_cellid: 'int[:, :]',
  cell_type: 'int[:]',
  tmp_cell_faces: 'int[:, :]',
  tmp_size_info: 'int[:]',
  tmp_cell_faces_map: 'int[:, :]',
  faces: 'int[:, :]',
  cell_faceid: 'int[:, :]',
  face_cellid: 'int[:, :]',
  cell_cellfid: 'int[:, :]',
  faces_counter: 'int[:]',
  boundary_info: 'int[:, :]' # array of size (nb_phy_face, 2)
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

  nb_faces = (tmp_cell_faces_map.shape[1] - 1) // 2
  for i in range(cells.shape[0]):
    _create_cell_faces(cells[i], tmp_cell_faces, tmp_size_info, cell_type[i])
    # For every face of the cell[i]
    # Get the intersection of the neighboring cells of this face's nodes (N*n*n)
    # The result should be two cells `intersect_cells`
    for j in range(tmp_size_info[-1]):
      _intersect_nodes(tmp_cell_faces[j], tmp_size_info[j], node_cellid, intersect_cells)
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
        for k in range(tmp_size_info[j]):
          faces[face_id, k] = tmp_cell_faces[j, k]
        faces[face_id, -1] = tmp_size_info[j]

        # Store the face in tmp_cell_faces_map for later existence verification.
        if intersect_cells[1] != -1:
          a = tmp_cell_faces_map[intersect_cells[1]]
          size = a[-1]
          a[size] = i
          a[nb_faces + size] = face_id
          a[-1] += 1

      # (cell_faces) Create cell faces
      cell_faceid[i, j] = face_id
      cell_faceid[i, -1] += 1

      # (face_cellid) Create neighboring cells of each face
      face_cellid[face_id, 0] = intersect_cells[0]
      face_cellid[face_id, 1] = intersect_cells[1]

      # (cell_cellfid) Create neighboring cells of the cell by face
      if intersect_cells[1] != -1:
        cell_cellfid[i, j] = intersect_cells[1]
        cell_cellfid[i, -1] += 1

  for i in range(len(cell_cellfid)):
    tmp = cell_cellfid[i]
    b = tmp[0:-1][tmp[0:-1] != -1]
    tmp[0:-1] = -1
    tmp[0:len(b)] = b

    b_counter = 0
    for j in range(tmp[-1]):
      face_id = tmp[j]

      # Create boundary if a face does not have a neighbor cell
      if face_cellid[face_id, 1] == -1:
        boundary_info[b_counter, 0] = i
        boundary_info[b_counter, 1] = j
        b_counter += 1

def _create_ghost_cells(cell_center: 'float[:, :]', face_center: 'float[:, :]', face_normal: 'float[:, :]', boundary_info: 'int[:, :]', ghost_info: 'int[:, :]'):
  # ghost_info => [c_x, c_y, c_z, cell_id, face_id]
  for i in range(len(boundary_info)):
    bc = boundary_info[i, 0] # boundary cell
    bf = boundary_info[i, 1] # boundary face

    f_center = face_center[bf]
    c_center = cell_center[bc]
    N = face_normal[bf]
    n_hat = N / np.linalg.norm(N)
    ghostcenter = c_center - 2 * np.dot(c_center - f_center, n_hat) * n_hat

    ghost_info[i, 0:3] = ghostcenter[0:3]
    ghost_info[i, 3] = bc
    ghost_info[i, 4] = bf


def _create_cellfid(
        cells: 'int[:, :]',
        node_cellid: 'int[:, :]',
        cell_type: 'int[:]',
        tmp_cell_faces: 'int[:, :, :]',
        tmp_size_info: 'int[:, :]',
        cell_cellfid: 'int[:, :]'
):
  intersect_cells = np.zeros(2, dtype=np.int32)

  for i in range(cells.shape[0]):
    _create_cell_faces(cells[i], tmp_cell_faces, tmp_size_info, cell_type[i])
    for j in range(tmp_size_info[-1]):
      _intersect_nodes(tmp_cell_faces[j], tmp_size_info[j], node_cellid, intersect_cells)
      # The face has at most two neighbors
      # swap to make intersect_cells[0] = cell_i id
      if intersect_cells[1] == i:
        intersect_cells[1] = intersect_cells[0]
        intersect_cells[0] = i

      if intersect_cells[1] != -1:
        size = cell_cellfid[i, -1]
        cell_cellfid[i, size] = intersect_cells[1]
        cell_cellfid[i, -1] += 1




# #############################################
# Face Name
# #############################################

def _get_face_name(
  phy_faces: 'int[:, :]',
  phy_faces_name: 'int[:]',
  face_nodes: 'int[:]',
  node_phyfaceid: 'int[:, :]',
):
  sorted_face_node = np.sort(face_nodes[0:-1])
  n = face_nodes[0] # select any node, choosing node 0
  for k in range(node_phyfaceid[n, -1]):
    f_index = node_phyfaceid[n, k]
    mesh_face = phy_faces[f_index][0:-1]
    mesh_face.sort()
    if np.all(mesh_face == sorted_face_node):
      return phy_faces_name[f_index]
  return -1

def get_max_node_faceid(faces: 'int[:, :]', arr: 'int[:]'):
  for i in range(len(faces)):
    for j in range(faces[i, -1]):
      n = faces[i, j]
      arr[n] += 1

def get_node_faceid(faces: 'int[:, :]', node_faceid: 'int[:, :]'):
  for i in range(len(faces)):
    for j in range(faces[i, -1]):
      n = faces[i, j]
      size = node_faceid[n, -1]
      node_faceid[n, size] = i
      node_faceid[n, -1] += 1

def define_face_and_node_name(
  phy_faces: 'int[:, :]',
  phy_faces_name: 'int[:]',
  faces: 'int[:, :]',
  node_phyfaceid: 'int[:, :]',
  face_haloid: 'int[:]',
  node_haloid: 'int[:, :]',
  face_oldname: 'int[:]',
  node_oldname: 'int[:]',
  face_name: 'int[:]',
  node_name: 'int[:]'
):
  for i in range(faces.shape[0]):
    name = _get_face_name(phy_faces, phy_faces_name, faces[i], node_phyfaceid)
    if name == -1:
      continue
    face_oldname[i] = name
    face_name[i] = name if face_haloid[i] == -1 else 10
    # Select the smallest name if it exists
    for j in range(faces[i, -1]):
      n = faces[i, j]
      if node_oldname[n] == 0 or node_oldname[n] > name:
        node_oldname[n] = name
        node_name[n] = name if node_haloid[n, -1] == 0 else 10




# #############################################
# Halos
# #############################################

def _create_halo_cells(cells, cell_faceid, faces, node_halos, node_haloid, cell_halofid, cell_halonid, face_haloid):
  nb_cells = len(cells)
  nb_faces = len(faces)
  intersect_cell = np.zeros(shape=2, dtype=np.int32)

  i = 0
  while i < len(node_halos):
    n = node_halos[i]
    c = node_halos[i + 1]
    for j in range(c):
      node_haloid[n, j] = node_halos[i + j + 2]
    node_haloid[n, -1] = c
    i += c + 2

  for i in range(nb_faces):
    nb_nodes = faces[i, -1]
    _intersect_nodes(faces[i, 0:nb_nodes], nb_nodes, node_haloid, intersect_cell)
    face_haloid[i] = intersect_cell[0]

  for i in range(nb_cells):
    for j in range(cells[i, -1]):
      n_halo = node_haloid[j]
      for k in range(n_halo[-1]):
        if _is_in_array(cell_halonid[i], n_halo[k]) == 0:
          size = cell_halonid[i, -1]
          cell_halonid[i, -1] += 1
          cell_halonid[i, size] = n_halo[k]
    for j in range(cell_faceid[i, -1]):
      face_id = cell_faceid[i, j]
      if face_haloid[face_id] != -1:
        size = cell_halofid[i, -1]
        cell_halofid[i, -1] += 1
        cell_halofid[i, size] = face_haloid[face_id]

def compile(func):
  #return func
  return numba.jit(nopython=True, fastmath=True, cache=True)(func)

# private
_is_in_array = compile(_is_in_array)
_binary_search = compile(_binary_search)
_intersect_nodes = compile(_intersect_nodes)
_create_cell_faces = compile(_create_cell_faces)
_polygon_area_2d = compile(_polygon_area_2d)
_tetrahedron_volume = compile(_tetrahedron_volume)
_triangle_area_3d = compile(_triangle_area_3d)
_triangle_normal_3d = compile(_triangle_normal_3d)
_get_face_name = compile(_get_face_name)

# public
append = compile(_append)
count_max_node_cellid = compile(_count_max_node_cellid)
create_node_cellid = compile(_create_node_cellid)
create_cellfid = compile(_create_cellfid)
count_max_cell_cellnid = compile(_count_max_cell_cellnid)
create_cell_cellnid = compile(_create_cell_cellnid)
create_info = compile(_create_info)
compute_cell_center_volume_2d = compile(_compute_cell_center_volume_2d)
compute_cell_center_volume_3d = compile(_compute_cell_center_volume_3d)
compute_face_info_2d = compile(_compute_face_info_2d)
compute_face_info_3d = compile(_compute_face_info_3d)
get_max_node_faceid = compile(get_max_node_faceid)
get_node_faceid = compile(get_node_faceid)
define_face_and_node_name = compile(define_face_and_node_name)
append_1d = compile(_append_1d)
create_halo_cells = compile(_create_halo_cells)
create_ghost_cells = compile(_create_ghost_cells)