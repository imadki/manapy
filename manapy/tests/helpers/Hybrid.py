import numpy as np

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

def _count_max_node_cellid(cells: 'int32[:, :]', res: 'int32[:]'):
  """
    Determine the max neighboring cells of a node across all cells
  """
  for cell in cells:
    for i in range(cell[-1]):
      node = cell[i]
      res[node] += 1


def _create_node_cellid(cells: 'int32[:, :]', node_cellid: 'int32[:, :]'):
  """
    Create neighboring cells for each node
  """
  for i in range(cells.shape[0]):
    for j in range(cells[i, -1]):
      node = node_cellid[cells[i, j]]
      size = node[-1]
      node[-1] += 1
      node[size] = i

  for i in range(node_cellid.shape[0]):
    node = node_cellid[i]
    node[0:node[-1]].sort()


def _count_max_cell_cellnid(cells: 'int32[:, :]', node_cellid: 'int32[:, :]', i_visited: 'int32[:]'):
  max_counter = 0
  for i in range(cells.shape[0]):
    counter = 0
    for j in range(cells[i][-1]):
      node_n = node_cellid[cells[i][j]]
      for k in range(node_n[-1]):
        if node_n[k] != i and i_visited[node_n[k]] != i:
          i_visited[node_n[k]] = i
          counter += 1
    max_counter = max(max_counter, counter)
  return max_counter


def _create_cell_cellnid(
        cells: 'int32[:, :]',
        node_cellid: 'int32[:, :]',
        cell_cellnid: 'int32[:, :]'
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


def _create_cell_faces(nodes: 'int32[:]', out_faces: 'int32[:, :]', size_info: 'int32[:]', cell_type: 'int32'):
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


def _create_info(
        cells: 'int32[:, :]',
        node_cellid: 'int32[:, :]',
        cell_type: 'int8[:]',
        tmp_cell_faces: 'int32[:, :]',
        tmp_size_info: 'int32[:]',
        tmp_cell_faces_map: 'int32[:, :]',
        faces: 'int32[:, :]',
        cell_faceid: 'int32[:, :]',
        face_cellid: 'int32[:, :]',
        cell_cellfid: 'int32[:, :]',
        faces_counter: 'int32[:]'
):
  """
    - Create faces
    - Create cells with their corresponding faces (cells.cellfid).
    - Create neighboring cells for each face (faces.cellid).
    - Create neighboring cells of a cell by face (cells.cellid).


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
        size = cell_cellfid[i, -1]
        # cell_cellfid[i, j] = intersect_cells[1]
        cell_cellfid[i, size] = intersect_cells[1]
        cell_cellfid[i, -1] += 1



def _set_cell_center(cells, nodes, cell_center):
  for i in range(len(cells)):
    points = nodes[cells[i, 0:cells[i, -1]]]
    cell_center[i, :] = np.sum(points, axis=0) / len(points)

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

def _set_cell_area(cells, nodes, cell_area, dim):
  if dim == 3:
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
      cell_area[i] = vol
  else:
    for i in range(len(cells)):
      nb_vertex = cells[i, -1]
      vertices = nodes[cells[i, 0:nb_vertex]]
      cell_area[i] = _polygon_area_2d(vertices)

# ##################################################
# ##################################################

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


def _compute_face_info_2d(faces: 'int32[:, :]', nodes: 'float[:, :]', face_cellid: 'int32[:, :]',
                          cell_center: 'float[:, :]', face_measure: 'float[:]', face_center: 'float[:, :]',
                          face_normal: 'float[:, :]'):
  for i in range(len(faces)):
    nb_vertex = faces[i, -1]
    points = nodes[faces[i, 0:nb_vertex], 0:2]

    # Face Measure
    u = points[0] - points[1]
    measure = np.sqrt(u[0] * u[0] + u[1] * u[1])
    face_measure[i] = measure

    # Center
    center = np.sum(points, axis=0) / nb_vertex
    face_center[i, 0:2] = center[0:2]

    # Face Normal
    normal = np.array([-u[1], u[0]], dtype=u.dtype)
    snorm = cell_center[face_cellid[i, 0], 0:2] - center
    if (np.dot(normal, snorm)) > 0:
      normal *= -1
    face_normal[i, 0:2] = normal


def _compute_face_info_3d(faces: 'int32[:, :]', nodes: 'float[:, :]', face_cellid: 'int32[:, :]',
                          cell_center: 'float[:, :]', face_measure: 'float[:]', face_center: 'float[:, :]',
                          face_normal: 'float[:, :]', face_tangent: 'float[:, :]', face_binormal: 'float[:, :]'):
  for i in range(len(faces)):
    nb_vertex = faces[i, -1]
    points = nodes[faces[i, 0:nb_vertex]]

    measure = 0
    normal = np.zeros(shape=3, dtype=points.dtype)
    if nb_vertex == 3:  # Triangle
      measure = _triangle_area_3d(points[0], points[1], points[2])
      normal[:] = _triangle_normal_3d(points[0], points[1], points[2])
    elif nb_vertex == 4:  # Rectangle
      measure = _triangle_area_3d(points[0], points[1], points[2]) + _triangle_area_3d(points[0], points[3], points[2])
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
    face_normal[i] = normal * 0.5  # 0.5 applicable to tetra and hexa

    # Calcul du binormal
    u = nodes[faces[i][1]] - nodes[faces[i][0]]
    face_tangent[i] = u
    face_binormal[i] = 0.5 * np.cross(u, normal)

# ##################################################
# ##################################################

def _create_node_phyid(phy_faces: 'int[:, :]', nb_nodes: 'int'):
  res = np.zeros(shape=nb_nodes, dtype=np.int32)
  _count_max_node_cellid(phy_faces, res)
  max_node_phyid = np.max(res)

  node_phyid = np.zeros(shape=(nb_nodes, max_node_phyid + 1), dtype=np.int32)
  _create_node_cellid(phy_faces, node_phyid)
  return node_phyid

def _define_node_oldname(phy_faces: 'int32[:, :]', phy_faces_name: 'int32[:]', node_oldname: 'int32[:]'):
  for i in range(phy_faces.shape[0]):
    name = phy_faces_name[i]
    for j in range(phy_faces[i, -1]):
      nodeid = phy_faces[i, j]
      # Select the smallest name if it exists
      if node_oldname[nodeid] == 0 or node_oldname[nodeid] > name:
        node_oldname[nodeid] = name

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

def _define_face_name(
        phy_faces: 'int32[:, :]',
        phy_faces_name: 'int32[:]',
        faces: 'int32[:, :]',
        node_phyfaceid: 'int32[:, :]',
        face_cellid: 'int32[:]',
        part_vert: 'int32[:]',
        face_oldname: 'int32[:]',
        face_name: 'int32[:]',
        phyid_to_faceid: 'int32[:]'
):
  for i in range(faces.shape[0]):
    phyid = _get_phyid(phy_faces, faces[i], node_phyfaceid)
    name = 0
    if phyid != -1:
      phyid_to_faceid[phyid] = i
      name = phy_faces_name[phyid]

    face_oldname[i] = name
    face_name[i] = name
    if face_cellid[i, 1] != -1 and part_vert[face_cellid[i, 0]] != part_vert[face_cellid[i, 1]]:
      face_name[i] = 10


# ##################################################
# ##################################################

def _get_ghost_tables_size(ghost_info: 'float[:, :]', faces: 'int32[:, :]', cell_faceid: 'int32[:, :]',
                           node_nb_ghostid: 'int32[:]'):
  for i in range(len(ghost_info)):
    bc = _reinterpret_float32_as_int32(ghost_info[i, 0])
    bf = _reinterpret_float32_as_int32(ghost_info[i, 1])
    fid = cell_faceid[bc, bf]
    for j in range(faces[fid, -1]):
      nid = faces[fid, j]
      node_nb_ghostid[nid] += 1

def _create_bf_cellid(phy_faces: 'int32[:, :]', node_cellid: 'int32[:, :]',
                      phyid_to_faceid: 'int32[:]', cell_faceid: 'int32[:, :]', intersect: 'int32[:]', bf_cellid: 'int32[:, :]'):
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


class HybridTestTables:
  def __init__(self, cell_loctoglob, cells, nodes, phy_faces, phy_faces_name, cell_type, max_cell_faceid, max_face_nodeid, dim):
    self.dim = dim
    self.cell_type = cell_type
    self.max_cell_faceid = max_cell_faceid
    self.max_face_nodeid = max_face_nodeid
    self.nb_cells = len(cells)
    self.nb_nodes = len(nodes)
    self.float_precision = 'float32'
    self.cells = cells
    self.nodes = nodes
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes)
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid)
    self.phy_faces = phy_faces
    self.phy_faces_name = phy_faces_name
    self.part_vert = self.get_part_vert(cell_loctoglob)
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid
    ) = self._create_info(self.cells, self.node_cellid, self.cell_type, self.max_cell_faceid, self.max_face_nodeid)
    self.nb_faces = len(self.faces)
    self.cell_center = self._set_cell_center(self.cells, self.nodes)
    self.cell_area = self._set_cell_area(self.cells, self.nodes)
    (
      self.face_measure,
      self.face_center,
      self.face_normal,
      self.face_tangent, # only in 3D, shape is 0 in 2D
      self.face_binormal # only in 3D, shape is 0 in 2D
    ) = self._create_face_info(self.faces, self.nodes, self.face_cellid, self.cell_center)
    (
      self.face_oldname,
      self.face_name,
      self.node_name,
      self.phyid_to_faceid
    ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_cellid, self.part_vert)

    self.ghost_info = self._create_shared_ghost_info(self.phy_faces, self.node_cellid, self.phyid_to_faceid, self.cell_center, self.cell_faceid, self.face_oldname, self.face_normal, self.face_center, self.face_measure)

  def get_part_vert(self, cell_loctoglob):
    part_vert = np.zeros(shape=self.nb_cells, dtype=np.int32)
    nb_partitions = len(cell_loctoglob)

    for p in range(nb_partitions):
      loctoglob = cell_loctoglob[p]
      for j in range(len(loctoglob)):
        global_index = loctoglob[j]
        part_vert[global_index] = p
    return part_vert

  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int'):
    # Count max node cellid
    res = np.zeros(shape=nb_nodes, dtype=np.int32)
    _count_max_node_cellid(cells, res)
    max_node_cellid = np.max(res)

    # Create node cellid
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32)
    _create_node_cellid(cells, node_cellid)
    return node_cellid

  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    # Count max cell cellnid
    i_visited = np.zeros(cells.shape[0], dtype=np.int32)
    max_cell_cellnid = _count_max_cell_cellnid(cells, node_cellid, i_visited)

    # Create cell cellnid
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=np.int32)
    _create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  def _set_cell_center(self, cells, nodes):
    cell_center = np.zeros(shape=(self.nb_cells, 3), dtype=self.float_precision)
    _set_cell_center(cells, nodes, cell_center)
    return cell_center

  def _set_cell_area(self, cells, nodes):
    cell_area = np.zeros(shape=self.nb_cells, dtype=self.float_precision)
    _set_cell_area(cells, nodes, cell_area, self.dim)
    return cell_area

  def _create_info(self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int'
  ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)
    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    faces_counter = np.zeros(shape=1, dtype=np.int32)

    _create_info(
      cells,
      node_cellid,
      cell_type,
      tmp_cell_faces,
      tmp_size_info,
      tmp_cell_faces_map,
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid,
      faces_counter
    )

    faces = faces[:faces_counter[0]]
    face_cellid = face_cellid[:faces_counter[0]]

    return (
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid
    )

  def _create_face_info(self, faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]'):
    nb_faces = len(faces)
    face_measure = np.zeros(shape=nb_faces, dtype=self.float_precision)
    face_center = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
    face_normal = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
    face_tangent = np.zeros(shape=0, dtype=self.float_precision)
    face_binormal = np.zeros(shape=0, dtype=self.float_precision)

    if self.dim == 2:
      _compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
      _compute_face_info_3d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal, face_tangent, face_binormal)
    return (
      face_measure,
      face_center,
      face_normal,
      face_tangent,
      face_binormal
    )

  def _define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 face_cellid: 'int[:, :]',
                                 part_vert: 'int[:]'
                                 ):
    nb_nodes = self.nb_nodes
    face_name = np.zeros(shape=faces.shape[0], dtype=np.int32)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=np.int32)
    phyid_to_faceid = np.ones(shape=phy_faces.shape[0], dtype=np.int32) * -1

    node_phyid = _create_node_phyid(phy_faces, nb_nodes)
    _define_face_name(phy_faces, phy_faces_name, faces, node_phyid, face_cellid, part_vert, face_oldname, face_name, phyid_to_faceid)

    node_oldname = np.zeros(shape=self.nb_nodes, dtype=np.int32)
    _define_node_oldname(phy_faces, phy_faces_name, node_oldname)

    return (
      face_oldname,
      face_name,
      node_oldname,
      phyid_to_faceid
    )


  def _create_shared_ghost_info(self, phy_faces, node_cellid, phyid_to_faceid, cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]'):

    intersect = np.zeros(shape=2, dtype=np.int32)
    bf_cellid = np.zeros(shape=(len(phy_faces), 2), dtype=np.int32)
    _create_bf_cellid(phy_faces, node_cellid, phyid_to_faceid, cell_faceid, intersect, bf_cellid)


    ghost_info_data_size = 13
    ghost_info = np.zeros(shape=(len(phy_faces), ghost_info_data_size), dtype=self.float_precision)

    _create_ghost_info_3d(bf_cellid, cell_center, cell_faceid, face_oldname, face_normal, face_center, face_measure, ghost_info)

    return ghost_info