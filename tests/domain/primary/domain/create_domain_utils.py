import numpy as np
import numba
from numba.typed import Dict
from numba import types

"""
###########
3D
##########
            
DONE::create_3d_halo_structure ->
  node_halonid renamed with node_haloid
  face_halofid
  node_name
  face_name
  face_oldname


DONE::face_info_3d ->
  cell_halonid [[halo_cells_by_node, ... size]] shape=(nb_cells, max_cell_cellnid + 1)
  face_ghostcenter [ghost_center x.y.z, gamma] shape=(nb_face, 4) dtype=float
  node_ghostcenter [[ghost_center x.y.z, cell_id, face_old_name, face_id]] shape=(nb_nodes, nb_node_ghost, 6) dtype=float
  node_ghostfaceinfo [[face_center x.y.z, face_normal x.y.z]] shape=(nb_nodes, nb_node_ghost, 6) dtype=float

update_pediodic_info_3d ?? It create 4 tables but it never return them
oriente_3dfacenodeid ?? Not sure what is the use case of orienting the faces, it could be oriented in the creation once for all

NOT_USED_ON_THE_PROJECTS::create_3doppNodeOfFaces

DONE::face_gradient_info_3d ->
  face_aitDiamond
  face_param1
  face_param2
  face_param3
  face_f_1
  face_f_2


DONE::create_info_3dfaces ->
  face_normal
  face_tangent
  face_binormal
  face_mesure
  face_center
  face_name
  

DONE::Compute_3dcentervolumeOfCell ->
  cell_center
  cell_volume
  
DONE::create_3dfaces ->
  faces
  cell_faceid

DONE::variables_3d ->
  nodes_R_x
  nodes_R_y
  nodes_R_z
  nodes_lambda_x
  nodes_lambda_y
  nodes_lambda_z
  nodes_number
  
######
2D
#####
DONE::create_cellsOfFace -> 
  face_cellid

DONE::create_cell_faceid ->
  cell_faceid

DONE::create_NeighborCellByFace ->
  cell_cellfid

DONE::create_node_cellid ->
  node_cellid 
  cell_cellnid 



DONE::create_node_ghostid -> ?
  node_ghostid [[faceid that has this ghost cell (node_ghostcenter[-1])]] shape=(nb_nodes, max_nghost + 1) 
  cell_ghostnid [[faceid that has this ghost cell (node_ghostcenter[-1])]] shape=(nb_cells, max_cell_nghost + 1)

DONE::face_info_2d -> 
  cell_halonid [[halo_cells_by_node, ... size]] shape=(nb_cells, max_cell_cellnid + 1)
  face_ghostcenter [ghost_center x.y, gamma] shape=(nb_face, 3) dtype=float
  node_ghostcenter [[ghost_center x.y, cell_id, face_old_name, face_id]] shape=(nb_nodes, nb_node_ghost, 5) dtype=float
  node_ghostfaceinfo [[face_center x.y, face_normal x.y]] shape=(nb_nodes, nb_node_ghost, 4) dtype=float


DONE::create_2d_halo_structure ->
  node_halonid
  face_halofid
  node_name
  face_name
  face_oldname

update_pediodic_info_2d -> the same thing it create tables and it does not return

DONE::Compute_2dcentervolumeOfCell ->
  cell_center
  cell_volume

DONE::create_2dfaces ->
  faces
  cell_faceid

DONE::create_info_2dfaces ->
  face_normal
  face_mesure
  face_center
  face_name
  
DONE::face_gradient_info_2d ->
  face.airDiamond
  face_param1
  face_param2
  face_param3
  face_param4
  face_f_1
  face_f_2
  face_f_3
  face_f_4
  
DONE::variables_2d ->
  nodes_R_x
  nodes_R_y
  nodes_lambda_x
  nodes_lambda_y
  nodes_number

DONE::create_NormalFacesOfCell ->
  cell_nf # normal face (only on 2D)
  
DONE::dist_ortho_function ->
  face_dist_ortho


DONE::_define_bounds:
  _bounds

DONE::_update_boundaries:
  _innerfaces
  _infaces
  _outfaces
  _upperfaces
  _bottomfaces
  _halofaces
  _boundaryfaces
  _periodicboundaryfaces
  _innernodes
  _innodes
  _outnodes
  _uppernodes
  _bottomnodes
  _halonodes
  _backfaces
  _frontfaces
  _frontnodes
  _backnodes
  _periodicboundarynodes
  _boundarynodes
  --
  _periodicinfaces
  _periodicoutfaces
  _periodicupperfaces
  _periodicbottomfaces
  _periodicinnodes
  _periodicoutnodes
  _periodicuppernodes
  _periodicbottomnodes
  _periodicbackfaces
  _periodicfrontfaces
  _periodicfrontnodes
  _periodicbacknodes






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
    for j in range(cells[i, -1]):
      node = node_cellid[cells[i, j]]
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

def _compute_face_info_3d(faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]', face_measure : 'float[:]', face_center: 'float[:, :]', face_normal: 'float[:, :]', face_tangent: 'float[:, :]', face_binormal:'float[:, :]'):
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

    # Calcul du binormal
    u = nodes[faces[i][1]] - nodes[faces[i][0]]
    face_tangent[i] = u
    face_binormal[i] = 0.5 * np.cross(u, normal)


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
  bf_cellid: 'int[:]',
  size: 'int'
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
  cmp = 0

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
        #cell_cellfid[i, j] = intersect_cells[1]
        cell_cellfid[i, size] = intersect_cells[1]
        cell_cellfid[i, -1] += 1
      elif size == 1:
        # works only if there is one partition
        if cmp == len(bf_cellid):
          raise RuntimeError("Number of physical faces does not match number of boundary faces !")
        bf_cellid[cmp, 0] = i
        bf_cellid[cmp, 1] = j
        cmp += 1

  # for i in range(len(cell_cellfid)):
  #   tmp = cell_cellfid[i]
  #   b = tmp[0:-1][tmp[0:-1] != -1]
  #   tmp[0:-1] = -1
  #   tmp[0:len(b)] = b


def _get_bf_recv_part_info(bf_recv_part_size: 'int[:]', rank: 'int', part_info: 'int[:]'):
  start = 0
  size = -1
  for i in range(0, len(bf_recv_part_size), 2):
    if bf_recv_part_size[i] == rank:
      size = bf_recv_part_size[i + 1]
      break
    start += bf_recv_part_size[i + 1]
  if size == -1:
    raise RuntimeError(f"Partition not found {bf_recv_part_size} {rank}")
  
  part_info[0] = start
  part_info[1] = size

# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################

def _create_ghost_info_2d(bf_cellid: 'int[:, :]', cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', ghost_info: 'float[:, :]', start: 'int'):
  # ghost_info [0=bc, 1=bf, 2=ghostcenter_x, 3=ghostcenter_y, 4=gamma, 5=face_oldname, 6=face_center_x, 7=face_center_y, 8=face_normal_x, 9=face_normal_y]

  cmp = start
  for i in range(len(bf_cellid)):
    cid = bf_cellid[i, 0]
    bf = bf_cellid[i, 1] # index of the face in the cell
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

    ghost_info[cmp, 0] = cid
    ghost_info[cmp, 1] = bf
    ghost_info[cmp, 2] = ghostcenter[0]
    ghost_info[cmp, 3] = ghostcenter[1]
    ghost_info[cmp, 4] = gamma
    ghost_info[cmp, 5] = face_oldname[fid]
    ghost_info[cmp, 6] = face_center[fid, 0]  # fc_x
    ghost_info[cmp, 7] = face_center[fid, 1]  # fc_y
    ghost_info[cmp, 8] = face_normal[fid, 0]  # fn_x
    ghost_info[cmp, 9] = face_normal[fid, 1]  # fn_y
    ghost_info[cmp, 10] = cell_loctoglob[cid] # global_id of local cell
    cmp += 1

def _create_ghost_info_3d(bf_cellid: 'int[:, :]', cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', ghost_info: 'float[:, :]', start: 'int'):
  # ghost_info [0=bc, 1=bf, 2=ghostcenter_x, 3=ghostcenter_y, 4=ghostcenter_z, 5=gamma, 6=face_oldname, 7=face_center_x, 8=face_center_y, 9=face_center_z, 10=face_normal_x, 11=face_normal_y, 12=face_normal_z]

  cmp = start
  for i in range(bf_cellid.shape[0]):
    cid = bf_cellid[i, 0]
    bf = bf_cellid[i, 1] # index of the face in the cell
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

    ghost_info[cmp, 0] = cid
    ghost_info[cmp, 1] = bf
    ghost_info[cmp, 2] = ghostcenter[0]
    ghost_info[cmp, 3] = ghostcenter[1]
    ghost_info[cmp, 4] = ghostcenter[2]
    ghost_info[cmp, 5] = gamma
    ghost_info[cmp, 6] = face_oldname[fid]
    ghost_info[cmp, 7] = face_center[fid, 0]  # fc_x
    ghost_info[cmp, 8] = face_center[fid, 1]  # fc_y
    ghost_info[cmp, 9] = face_center[fid, 2]  # fc_z
    ghost_info[cmp, 10] = face_normal[fid, 0]  # fn_x
    ghost_info[cmp, 11] = face_normal[fid, 1]  # fn_y
    ghost_info[cmp, 12] = face_normal[fid, 2]  # fn_z
    if cell_loctoglob.shape[0] != 0:
      ghost_info[cmp, 13] = cell_loctoglob[cid] # global_id of local cell
    cmp += 1

def _get_ghost_tables_size(ghost_info: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', node_nb_ghostid: 'int[:]', start: 'int', end: 'int'):

  for i in range(start, end):
    bc = int(ghost_info[i, 0])
    bf = int(ghost_info[i, 1])
    fid = cell_faceid[bc, bf]
    for j in range(faces[fid, -1]):
      nid = faces[fid, j]
      node_nb_ghostid[nid] += 1


def _create_ghost_tables_2d(ghost_info: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', node_ghostid: 'int[:, :]', node_ghostcenter: 'int[:, :]', face_ghostcenter: 'int[:, :]', node_ghostfaceinfo: 'int[:, :]', start: 'int', end: 'int'):
  # node_ghostid
  # node_ghostcenter
  # face_ghostcenter
  # node_ghostfaceinfo

  for i in range(start, end):
    bc = int(ghost_info[i, 0])
    bf = int(ghost_info[i, 1])
    ghostcenter_x = ghost_info[i, 2]
    ghostcenter_y = ghost_info[i, 3]
    gamma = ghost_info[i, 4]
    face_oldname = ghost_info[i, 5]
    face_center_x = ghost_info[i, 6]
    face_center_y = ghost_info[i, 7]
    face_normal_x = ghost_info[i, 8]
    face_normal_y = ghost_info[i, 9]

    fid = cell_faceid[bc, bf]

    # face_ghostcenter
    face_ghostcenter[fid, 0] = ghostcenter_x
    face_ghostcenter[fid, 1] = ghostcenter_y
    face_ghostcenter[fid, 2] = gamma

    # node_ghostid, node_ghostcenter, node_ghostfaceinfo
    for j in range(faces[fid, -1]):
      nid = faces[fid, j]
      size = node_ghostid[nid, -1]
      node_ghostid[nid, -1] += 1
      node_ghostid[nid, size] = fid # face_id

      node_ghostcenter[nid, size, 0] = ghostcenter_x
      node_ghostcenter[nid, size, 1] = ghostcenter_y
      node_ghostcenter[nid, size, 2] = bc
      node_ghostcenter[nid, size, 3] = face_oldname # face_old_name
      node_ghostcenter[nid, size, 4] = fid

      node_ghostfaceinfo[nid, size, 0] = face_center_x
      node_ghostfaceinfo[nid, size, 1] = face_center_y
      node_ghostfaceinfo[nid, size, 2] = face_normal_x
      node_ghostfaceinfo[nid, size, 3] = face_normal_y

def _create_ghost_tables_3d(ghost_info: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', node_ghostid: 'int[:, :]', node_ghostcenter: 'int[:, :]', face_ghostcenter: 'int[:, :]', node_ghostfaceinfo: 'int[:, :]', start: 'int', end: 'int'):

  for i in range(start, end):
    bc = int(ghost_info[i, 0])
    bf = int(ghost_info[i, 1])
    ghostcenter_x = ghost_info[i, 2]
    ghostcenter_y = ghost_info[i, 3]
    ghostcenter_z = ghost_info[i, 4]
    gamma = ghost_info[i, 5]
    face_oldname = ghost_info[i, 6]
    face_center_x = ghost_info[i, 7]
    face_center_y = ghost_info[i, 8]
    face_center_z = ghost_info[i, 9]
    face_normal_x = ghost_info[i, 10]
    face_normal_y = ghost_info[i, 11]
    face_normal_z = ghost_info[i, 12]

    fid = cell_faceid[bc, bf]


    # face_ghostcenter
    face_ghostcenter[fid, 0] = ghostcenter_x
    face_ghostcenter[fid, 1] = ghostcenter_y
    face_ghostcenter[fid, 2] = ghostcenter_z
    face_ghostcenter[fid, 3] = gamma

    # node_ghostid, node_ghostcenter, node_ghostfaceinfo
    for j in range(faces[fid, -1]):
      nid = faces[fid, j]
      size = node_ghostid[nid, -1]
      node_ghostid[nid, -1] += 1
      node_ghostid[nid, size] = fid # face_id

      node_ghostcenter[nid, size, 0] = ghostcenter_x
      node_ghostcenter[nid, size, 1] = ghostcenter_y
      node_ghostcenter[nid, size, 2] = ghostcenter_z
      node_ghostcenter[nid, size, 3] = bc
      node_ghostcenter[nid, size, 4] = face_oldname # face_old_name
      node_ghostcenter[nid, size, 5] = fid

      node_ghostfaceinfo[nid, size, 0] = face_center_x
      node_ghostfaceinfo[nid, size, 1] = face_center_y
      node_ghostfaceinfo[nid, size, 2] = face_center_z
      node_ghostfaceinfo[nid, size, 3] = face_normal_x
      node_ghostfaceinfo[nid, size, 4] = face_normal_y
      node_ghostfaceinfo[nid, size, 5] = face_normal_z


def _get_cell_ghostnid_size(cells: 'int[:, :]', node_ghostid: 'int[:, :]',
                            bc_visited: 'int[:]', ghost_visited: 'int[:]', cell_ghostnid_size: 'int[:]'):
  for i in range(len(cells)):
    if bc_visited[i] == 1:
      continue
    bc_visited[i] = 1
    for j in range(cells[i, -1]):
      nid = cells[i, j]
      for k in range(node_ghostid[nid, -1]):
        g_id = node_ghostid[nid, k]
        if ghost_visited[g_id] != i:
          ghost_visited[g_id] = i
          cell_ghostnid_size[i] += 1


def _create_cell_ghostnid(cells: 'int[:, :]', node_ghostid: 'int[:, :]', bc_visited: 'int[:]', ghost_visited: 'int[:]', cell_ghostnid: 'int[:, :]'):
  for i in range(len(cells)):
    if bc_visited[i] == 1:
      continue
    bc_visited[i] = 1
    for j in range(cells[i, -1]):
      nid = cells[i, j]
      for k in range(node_ghostid[nid, -1]):
        g_id = node_ghostid[nid, k]
        if ghost_visited[g_id] != i:
          ghost_visited[g_id] = i
          size = cell_ghostnid[i, -1]
          cell_ghostnid[i, -1] += 1
          cell_ghostnid[i, size] = g_id

def _count_max_bcell_halobfid(cells: 'int[:, :]', b_ncellid: 'int[:, :]', node_halobfid: 'int[:, :]', visited: 'int[:]'):

  max_counter = 0
  for i in range(b_ncellid.shape[0]):
    bc = b_ncellid[i]
    cell = cells[bc]
    counter = 0
    for j in range(cell[-1]):
      node_nbf = node_halobfid[cell[j]]
      for k in range(node_nbf[-1]):
        if visited[node_nbf[k]] != i:
          visited[node_nbf[k]] = i
          counter += 1
    max_counter = max(max_counter, counter)
  return max_counter


def _create_bcell_halobfid(cells: 'int[:, :]', b_ncellid: 'int[:, :]', node_halobfid: 'int[:, :]', visited: 'int[:]', bcell_halobfid: 'int[:, :]'):

  for i in range(b_ncellid.shape[0]):
    bc = b_ncellid[i]
    cell = cells[bc]

    bcell_halobfid[i, 0] = bc
    counter = 1
    for j in range(cell[-1]):
      node_nbf = node_halobfid[cell[j]]
      for k in range(node_nbf[-1]):
        if visited[node_nbf[k]] != i:
          visited[node_nbf[k]] = i
          bcell_halobfid[i, counter] = node_nbf[k]
          counter += 1
    bcell_halobfid[i, -1] = counter

def _count_max_b_nodeid(phy_faces: 'int[:, :]', visited: 'int[:]'):

  counter = 0
  for i in range(phy_faces.shape[0]):
    for j in range(phy_faces[i, -1]):
      bn = phy_faces[i, j]
      if visited[bn] == 0:
        visited[bn] = 1
        counter += 1
  return counter

def _create_b_nodeid(phy_faces: 'int[:, :]', visited: 'int[:]', b_nodeid: 'int[:]'):

  counter = 0
  for i in range(phy_faces.shape[0]):
    for j in range(phy_faces[i, -1]):
      bn = phy_faces[i, j]
      if visited[bn] == 0:
        visited[bn] = 1
        b_nodeid[counter] = bn
        counter += 1
  return counter


def _get_max_b_ncellid(b_nodeid, node_cellid, visited):
  cmp = 0
  for i in range(b_nodeid.shape[0]):
    nodeid = b_nodeid[i]
    for j in range(node_cellid[nodeid, -1]):
      cell = node_cellid[nodeid, j]
      if visited[cell] == 0:
        visited[cell] = 1
        cmp += 1
  return cmp

def _create_b_ncellid(b_nodeid, node_cellid, visited, b_ncellid):
  cmp = 0
  for i in range(b_nodeid.shape[0]):
    nodeid = b_nodeid[i]
    for j in range(node_cellid[nodeid, -1]):
      cell = node_cellid[nodeid, j]
      if visited[cell] == 0:
        visited[cell] = 1
        b_ncellid[cmp] = cell
        cmp += 1


def _create_ghost_new_index(ghost_part_size: 'int[:]', ghost_new_index: 'int[:]'):
  start = ghost_part_size[0]
  end = start + ghost_part_size[1]
  cmp = 0
  for i in range(len(ghost_new_index)):
    if start <= i < end:
      ghost_new_index[i] = -1
    else:
      ghost_new_index[i] = cmp
      cmp += 1
  return cmp

def _search_halo_cell(node_halo_cells: 'int[:]', halo_haloext: 'int[:, :]', item: 'int'):
  # return an index point to halo_haloext of the cell global id referred by item
  # item is the global index of a neighbor cell
  for i in range(node_halo_cells[-1]):
    # for every neighbor halo cell
    n_halo_cell = node_halo_cells[i]
    g_index = halo_haloext[n_halo_cell, 0] # get global if of the halocell
    if g_index == item:
      return n_halo_cell
  raise RuntimeError(f"{item} must be in halo_haloext of node_halo_cells {node_halo_cells}")

def _create_halo_ghost_tables_2d(ghost_info: 'int[:, :]', bcell_halobfid: 'int[:, :]', b_nodeid: 'int[:]', node_halobfid: 'int[:, :]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', ghost_new_index: 'int[:]', cell_haloghostnid: 'int[:, :]', cell_haloghostcenter: 'float[:, :]', node_haloghostid: 'int[:, :]', node_haloghostcenter: 'float[:, :, :]', node_haloghostfaceinfo: 'float[:, :, :]'):
  """
  * cell_haloghostcenter [[g_x, g_y]]
  * cell_haloghostnid [[indices point to cell_haloghostcenter]]
  * node_haloghostid [[indices point to cell_haloghostcenter]]
  * node_haloghostcenter [[[g_x, g_y, (halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
  * node_haloghostfaceinfo [[[fc_x, fc_y, fn_x, fn_y]]]


  """
  for i in range(len(bcell_halobfid)):
    bc = bcell_halobfid[i, 0]
    for j in range(1, bcell_halobfid[i, -1]):
      hg_id = bcell_halobfid[i, j]

      cell_haloghostcenter[ghost_new_index[hg_id], 0] = ghost_info[hg_id, 2] # g_x
      cell_haloghostcenter[ghost_new_index[hg_id], 1] = ghost_info[hg_id, 3] # g_y

      cell_haloghostnid[bc, j - 1] = ghost_new_index[hg_id]
    cell_haloghostnid[bc, -1] = bcell_halobfid[i, -1] - 1


  for i in range(len(b_nodeid)):
    bn = b_nodeid[i]
    for j in range(node_halobfid[bn, -1]):
      hg_id = node_halobfid[bn, j] # halo_ghost_index
      node_haloghostid[bn, j] = ghost_new_index[hg_id]

      cell_globalid = np.int32(ghost_info[hg_id, 10])
      node_haloghostcenter[bn, j, 0] = ghost_info[hg_id, 2] # g_x
      node_haloghostcenter[bn, j, 1] = ghost_info[hg_id, 3] # g_y
      node_haloghostcenter[bn, j, 2] = _search_halo_cell(node_haloid[bn], halo_halosext, cell_globalid) # index point to halo_halosext of cell_globalid
      node_haloghostcenter[bn, j, 3] = ghost_info[hg_id, 5] #face_old_name
      node_haloghostcenter[bn, j, 4] = ghost_new_index[hg_id] # index point to cell_haloghostcenter

      node_haloghostfaceinfo[bn, j, 0] = ghost_info[hg_id, 6]
      node_haloghostfaceinfo[bn, j, 1] = ghost_info[hg_id, 7]
      node_haloghostfaceinfo[bn, j, 2] = ghost_info[hg_id, 8]
      node_haloghostfaceinfo[bn, j, 3] = ghost_info[hg_id, 9]
    node_haloghostid[bn, -1] = node_halobfid[bn, -1]

def _create_halo_ghost_tables_3d(ghost_info: 'int[:, :]', bcell_halobfid: 'int[:, :]', b_nodeid: 'int[:]', node_halobfid: 'int[:, :]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', ghost_new_index: 'int[:]', cell_haloghostnid: 'int[:, :]', cell_haloghostcenter: 'float[:, :]', node_haloghostid: 'int[:, :]', node_haloghostcenter: 'float[:, :, :]', node_haloghostfaceinfo: 'float[:, :, :]'):
  """
  * cell_haloghostcenter [[g_x, g_y, g_z]]
  * cell_haloghostnid [[indices point to cell_haloghostcenter]]
  * node_haloghostid [[indices point to cell_haloghostcenter]]
  * node_haloghostcenter [[[g_x, g_y, g_z, (halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
  * node_haloghostfaceinfo [[[fc_x, fc_y, fc_z, fn_x, fn_y, fn_z]]]


  """
  for i in range(len(bcell_halobfid)):
    bc = bcell_halobfid[i, 0]
    for j in range(1, bcell_halobfid[i, -1]):
      hg_id = bcell_halobfid[i, j]

      cell_haloghostcenter[ghost_new_index[hg_id], 0] = ghost_info[hg_id, 2] # g_x
      cell_haloghostcenter[ghost_new_index[hg_id], 1] = ghost_info[hg_id, 3] # g_y
      cell_haloghostcenter[ghost_new_index[hg_id], 2] = ghost_info[hg_id, 4] # g_z

      cell_haloghostnid[bc, j - 1] = ghost_new_index[hg_id]
    cell_haloghostnid[bc, -1] = bcell_halobfid[i, -1] - 1


  for i in range(len(b_nodeid)):
    bn = b_nodeid[i]
    for j in range(node_halobfid[bn, -1]):
      hg_id = node_halobfid[bn, j] # halo_ghost_index
      node_haloghostid[bn, j] = ghost_new_index[hg_id]

      cell_globalid = np.int32(ghost_info[hg_id, 13])
      node_haloghostcenter[bn, j, 0] = ghost_info[hg_id, 2] # g_x
      node_haloghostcenter[bn, j, 1] = ghost_info[hg_id, 3] # g_y
      node_haloghostcenter[bn, j, 2] = ghost_info[hg_id, 4] # g_z
      node_haloghostcenter[bn, j, 3] = _search_halo_cell(node_haloid[bn], halo_halosext, cell_globalid) # index point to halo_halosext of cell_globalid
      node_haloghostcenter[bn, j, 4] = ghost_info[hg_id, 6] #face_old_name
      node_haloghostcenter[bn, j, 5] = ghost_new_index[hg_id] # index point to cell_haloghostcenter

      node_haloghostfaceinfo[bn, j, 0] = ghost_info[hg_id, 7]
      node_haloghostfaceinfo[bn, j, 1] = ghost_info[hg_id, 8]
      node_haloghostfaceinfo[bn, j, 2] = ghost_info[hg_id, 9]
      node_haloghostfaceinfo[bn, j, 3] = ghost_info[hg_id, 10]
      node_haloghostfaceinfo[bn, j, 4] = ghost_info[hg_id, 11]
      node_haloghostfaceinfo[bn, j, 5] = ghost_info[hg_id, 12]
    node_haloghostid[bn, -1] = node_halobfid[bn, -1]



def _create_cellfid_and_bf_info(
        cells: 'int[:, :]',
        node_cellid: 'int[:, :]',
        cell_type: 'int[:]',
        tmp_cell_faces: 'int[:, :]',
        tmp_size_info: 'int[:]',
        cell_cellfid: 'int[:, :]',
        bf_cellid: 'int[:, :]',
        bf_nodes: 'int[:, :]'
):
  intersect_cells = np.zeros(2, dtype=np.int32)

  cmp = 0
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
      else:
        # bf_cellid, bf_nodes has the same size of phy_faces
        if cmp == len(bf_cellid):
          raise RuntimeError("Number of physical faces does not match number of boundary faces !")
        bf_cellid[cmp, 0] = i
        bf_cellid[cmp, 1] = j
        for k in range(tmp_size_info[j]):
          bf_nodes[cmp, k] = tmp_cell_faces[j, k]
        bf_nodes[cmp, -1] = tmp_size_info[j]
        cmp += 1



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

def _get_max_node_faceid(faces: 'int[:, :]', arr: 'int[:]'):
  for i in range(len(faces)):
    for j in range(faces[i, -1]):
      n = faces[i, j]
      arr[n] += 1

def _get_node_faceid(faces: 'int[:, :]', node_faceid: 'int[:, :]'):
  for i in range(len(faces)):
    for j in range(faces[i, -1]):
      n = faces[i, j]
      size = node_faceid[n, -1]
      node_faceid[n, size] = i
      node_faceid[n, -1] += 1

def _define_face_and_node_name(
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
    has_haloid = not (face_haloid.shape[0] == 0 or face_haloid[i] == -1)
    face_name[i] = 10 if has_haloid else name
    # Select the smallest name if it exists
    for j in range(faces[i, -1]):
      n = faces[i, j]
      if node_oldname[n] == 0 or node_oldname[n] > name:
        node_oldname[n] = name
        has_haloid = not (node_haloid.shape[0] == 0 or node_haloid[n, -1])
        node_name[n] = 10 if has_haloid else name




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
      node = cells[i, j]
      n_halo = node_haloid[node]
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

# The same as the original
def _face_gradient_info_2d(face_cellid: 'int[:,:]', faces: 'int[:,:]', face_ghostcenter: 'float[:,:]', face_name: 'int[:]', face_normal: 'float[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', face_haloid: 'int[:]', nodes: 'float[:,:]', face_air_diamond: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_param4: 'float[:]', face_f1: 'float[:,:]', face_f2: 'float[:,:]', face_f3: 'float[:,:]', face_f4: 'float[:,:]', cell_shift: 'float[:,:]'):

  nbface = len(face_cellid)

  dim = 2
  v_2 = np.zeros(dim, dtype=face_air_diamond.dtype)

  for i in range(nbface):

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    xy_1 = nodes[i_1][0:dim]
    xy_2 = nodes[i_2][0:dim]

    v_1 = cell_center[c_left][0:dim]

    if face_name[i] == 0:
      v_2[:] = cell_center[c_right][0:dim]
    elif face_name[i] == 11 or face_name[i] == 22:
      v_2[0] = cell_center[c_right][0] + cell_shift[c_right][0]
      v_2[1] = cell_center[c_right][1]
    elif face_name[i] == 33 or face_name[i] == 44:
      v_2[0] = cell_center[c_right][0]
      v_2[1] = cell_center[c_right][1] + cell_shift[c_right][1]
    elif face_name[i] == 10:
      v_2[:] = halo_centvol[face_haloid[i]][0:dim]
    else:
      v_2[:] = face_ghostcenter[i][0:dim]

    face_f1[i][:] = v_1[:] - xy_1[:]
    face_f2[i][:] = xy_2[:] - v_1[:]
    face_f3[i][:] = v_2[:] - xy_2[:]
    face_f4[i][:] = xy_1[:] - v_2[:]

    n1 = face_normal[i][0]
    n2 = face_normal[i][1]

    face_air_diamond[i] = 0.5 * ((xy_2[0] - xy_1[0]) * (v_2[1] - v_1[1]) + (v_1[0] - v_2[0]) * (xy_2[1] - xy_1[1]))

    # TODO check why face_air_diamond is 0
    if face_air_diamond[i] == 0:
      return

    face_param1[i] = 1. / (2. * face_air_diamond[i]) * ((face_f1[i][1] + face_f2[i][1]) * n1 - (face_f1[i][0] + face_f2[i][0]) * n2)
    face_param2[i] = 1. / (2. * face_air_diamond[i]) * ((face_f2[i][1] + face_f3[i][1]) * n1 - (face_f2[i][0] + face_f3[i][0]) * n2)
    face_param3[i] = 1. / (2. * face_air_diamond[i]) * ((face_f3[i][1] + face_f4[i][1]) * n1 - (face_f3[i][0] + face_f4[i][0]) * n2)
    face_param4[i] = 1. / (2. * face_air_diamond[i]) * ((face_f4[i][1] + face_f1[i][1]) * n1 - (face_f4[i][0] + face_f1[i][0]) * n2)

# The same as the original
def _face_gradient_info_3d(face_cellid: 'int[:,:]', faces: 'int[:,:]', face_ghostcenter: 'float[:,:]', face_name: 'int[:]', face_normal: 'float[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', face_haloid: 'int[:]', nodes: 'float[:,:]', face_air_diamond: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_f1: 'float[:,:]', face_f2: 'float[:,:]', cell_shift: 'float[:,:]'):

  nbfaces = len(face_cellid)

  dim = 3
  v_2 = np.zeros(dim, dtype=face_air_diamond.dtype)

  for i in range(nbfaces):

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    v_1 = cell_center[c_left][0:dim]

    if face_name[i] == 0:
      v_2[:] = cell_center[c_right][0:dim]

    elif face_name[i] == 11 or face_name[i] == 22:
      v_2[0] = cell_center[c_right][0] + cell_shift[c_right][0]
      v_2[1] = cell_center[c_right][1]
      v_2[2] = cell_center[c_right][2]
    elif face_name[i] == 33 or face_name[i] == 44:
      v_2[0] = cell_center[c_right][0]
      v_2[1] = cell_center[c_right][1] + cell_shift[c_right][1]
      v_2[2] = cell_center[c_right][2]
    elif face_name[i] == 55 or face_name[i] == 66:
      v_2[0] = cell_center[c_right][0]
      v_2[1] = cell_center[c_right][1]
      v_2[2] = cell_center[c_right][2] + cell_shift[c_right][2]

    elif face_name[i] == 10:
      v_2[:] = halo_centvol[face_haloid[i]][0:dim]
    else:
      v_2[:] = face_ghostcenter[i][0:dim]

    s1 = v_2 - nodes[i_2][0:dim]
    s2 = nodes[i_4][0:dim] - nodes[i_2][0:dim]
    s3 = v_1 - nodes[i_2][0:dim]
    face_f1[i][:] = (0.5 * np.cross(s1, s2)) + (0.5 * np.cross(s2, s3))

    s4 = v_2 - nodes[i_3][0:dim]
    s5 = nodes[i_1][0:dim] - nodes[i_3][0:dim]
    s6 = v_1 - nodes[i_3][0:dim]
    face_f2[i][:] = (0.5 * np.cross(s4, s5)) + (0.5 * np.cross(s5, s6))

    s7 = v_2 - v_1
    face_air_diamond[i] = np.dot(face_normal[i], s7)

    # TODO check why face_air_diamond is 0
    if face_air_diamond[i] == 0:
      return

    face_param1[i] = np.dot(face_f1[i], face_normal[i]) / face_air_diamond[i]
    face_param2[i] = np.dot(face_f2[i], face_normal[i]) / face_air_diamond[i]
    face_param3[i] = np.dot(face_normal[i], face_normal[i]) / face_air_diamond[i]

# tables are Modified
def _variables_2d(cell_center: 'float[:,:]', node_cellid: 'int[:,:]', node_haloid: 'int[:,:]', node_ghostid: 'int[:,:]', node_haloghostid: 'int[:,:]', node_periodicid: 'int[:,:]', nodes: 'float[:,:]', node_oldname: 'int[:, :]', face_ghostcenter: 'float[:,:]', cell_haloghostcenter: 'float[:,:]', halo_centvol: 'float[:,:]', node_R_x: 'float[:]', node_R_y: 'float[:]', node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_number: 'int[:]', cell_shift: 'float[:,:]'):

  nbnode = len(node_R_x)

  for i in range(nbnode):
    I_xx = 0.0
    I_yy = 0.0
    I_xy = 0.0

    for j in range(node_cellid[i][-1]):
      center = cell_center[node_cellid[i][j]][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_xy += (Rx * Ry)
      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_number[i] += 1

    for j in range(node_ghostid[i][-1]):
      center = face_ghostcenter[node_ghostid[i][j]][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_xy += (Rx * Ry)
      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_number[i] += 1

    # periodic boundary old vertex names)
    if node_oldname[i] == 11 or node_oldname[i] == 22:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center_x = cell_center[cell][0] + cell_shift[cell][0]
        center_y = cell_center[cell][1]

        Rx = center_x - nodes[i][0]
        Ry = center_y - nodes[i][1]
        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_xy += (Rx * Ry)
        node_R_x[i] += Rx
        node_R_y[i] += Ry
        node_number[i] += 1

    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center_x = cell_center[cell][0]
        center_y = cell_center[cell][1] + cell_shift[cell][1]

        Rx = center_x - nodes[i][0]
        Ry = center_y - nodes[i][1]
        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_xy += (Rx * Ry)
        node_R_x[i] += Rx
        node_R_y[i] += Ry
        node_number[i] += 1

    for j in range(node_haloghostid[i][-1]):
      cell = node_haloghostid[i][j]

      center = cell_haloghostcenter[cell]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]

      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_xy += (Rx * Ry)
      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_number[i] = node_number[i] + 1

      # if haloidn[i][-1] > 0:
    for j in range(node_haloid[i][-1]):
      cell = node_haloid[i][j]
      center = halo_centvol[cell][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_xy += (Rx * Ry)
      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_number[i] = node_number[i] + 1

    D = I_xx * I_yy - I_xy * I_xy

    # TODO check why D is 0
    if D == 0.0:
      return
    node_lambda_x[i] = (I_xy * node_R_y[i] - I_yy * node_R_x[i]) / D
    node_lambda_y[i] = (I_xy * node_R_x[i] - I_xx * node_R_y[i]) / D


def _variables_3d(cell_center: 'float[:,:]', node_cellid: 'int[:,:]', node_haloid: 'int[:,:]', node_ghostid: 'int[:,:]', node_haloghostid: 'int[:,:]', node_periodicid: 'int[:,:]', nodes: 'float[:,:]', node_oldname: 'int[:, :]', face_ghostcenter: 'float[:,:]', cell_haloghostcenter: 'float[:,:]', halo_centvol: 'float[:,:]', node_R_x: 'float[:]', node_R_y: 'float[:]', node_R_z: 'float[:]', node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'int[:]', cell_shift: 'float[:,:]'):

  nbnode = len(node_R_x)

  for i in range(nbnode):
    I_xx = 0.0
    I_yy = 0.0
    I_zz = 0.0
    I_xy = 0.0
    I_xz = 0.0
    I_yz = 0.0

    for j in range(node_cellid[i][-1]):
      center = cell_center[node_cellid[i][j]][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      Rz = center[2] - nodes[i][2]

      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_zz += (Rz * Rz)
      I_xy += (Rx * Ry)
      I_xz += (Rx * Rz)
      I_yz += (Ry * Rz)

      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_R_z[i] += Rz

      node_number[i] += 1

    for j in range(node_ghostid[i][-1]):
      center = face_ghostcenter[node_ghostid[i][j]][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      Rz = center[2] - nodes[i][2]

      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_zz += (Rz * Rz)
      I_xy += (Rx * Ry)
      I_xz += (Rx * Rz)
      I_yz += (Ry * Rz)

      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_R_z[i] += Rz

      node_number[i] += 1

    # periodic boundary old vertex names)
    if node_oldname[i] == 11 or node_oldname[i] == 22:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center_x = cell_center[cell][0] + cell_shift[cell][0]
        center_y = cell_center[cell][1]
        center_z = cell_center[cell][2]

        Rx = center_x - nodes[i][0]
        Ry = center_y - nodes[i][1]
        Rz = center_z - nodes[i][2]

        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_zz += (Rz * Rz)
        I_xy += (Rx * Ry)
        I_xz += (Rx * Rz)
        I_yz += (Ry * Rz)

        node_R_x[i] += Rx
        node_R_y[i] += Ry
        node_R_z[i] += Rz
        node_number[i] = node_number[i] + 1

    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center_x = cell_center[cell][0]
        center_y = cell_center[cell][1] + cell_shift[cell][1]
        center_z = cell_center[cell][2]

        Rx = center_x - nodes[i][0]
        Ry = center_y - nodes[i][1]
        Rz = center_z - nodes[i][2]

        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_zz += (Rz * Rz)
        I_xy += (Rx * Ry)
        I_xz += (Rx * Rz)
        I_yz += (Ry * Rz)

        node_R_x[i] += Rx
        node_R_y[i] += Ry
        node_R_z[i] += Rz
        node_number[i] = node_number[i] + 1

    elif node_oldname[i] == 55 or node_oldname[i] == 66:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center_x = cell_center[cell][0]
        center_y = cell_center[cell][1]
        center_z = cell_center[cell][2] + cell_shift[cell][2]

        Rx = center_x - nodes[i][0]
        Ry = center_y - nodes[i][1]
        Rz = center_z - nodes[i][2]

        I_xx += (Rx * Rx)
        I_yy += (Ry * Ry)
        I_zz += (Rz * Rz)
        I_xy += (Rx * Ry)
        I_xz += (Rx * Rz)
        I_yz += (Ry * Rz)

        node_R_x[i] += Rx
        node_R_y[i] += Ry
        node_R_z[i] += Rz
        node_number[i] = node_number[i] + 1

    for j in range(node_haloid[i][-1]):
      cell = node_haloid[i][j]
      center = halo_centvol[cell][0:3]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      Rz = center[2] - nodes[i][2]

      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_zz += (Rz * Rz)
      I_xy += (Rx * Ry)
      I_xz += (Rx * Rz)
      I_yz += (Ry * Rz)

      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_R_z[i] += Rz
      node_number[i] = node_number[i] + 1

    for j in range(node_haloghostid[i][-1]):
      cell = node_haloghostid[i][j]
      center = cell_haloghostcenter[cell]
      Rx = center[0] - nodes[i][0]
      Ry = center[1] - nodes[i][1]
      Rz = center[2] - nodes[i][2]

      I_xx += (Rx * Rx)
      I_yy += (Ry * Ry)
      I_zz += (Rz * Rz)
      I_xy += (Rx * Ry)
      I_xz += (Rx * Rz)
      I_yz += (Ry * Rz)

      node_R_x[i] += Rx
      node_R_y[i] += Ry
      node_R_z[i] += Rz
      node_number[i] = node_number[i] + 1

    D = I_xx * I_yy * I_zz + 2 * I_xy * I_xz * I_yz - I_xx * I_yz * I_yz - I_yy * I_xz * I_xz - I_zz * I_xy * I_xy

    # TODO check why D is 0
    if D==0.0:
      return

    node_lambda_x[i] = ((I_yz * I_yz - I_yy * I_zz) * node_R_x[i] + (I_xy * I_zz - I_xz * I_yz) * node_R_y[i] + (I_xz * I_yy - I_xy * I_yz) * node_R_z[i]) / D
    node_lambda_y[i] = ((I_xy * I_zz - I_xz * I_yz) * node_R_x[i] + (I_xz * I_xz - I_xx * I_zz) * node_R_y[i] + (I_yz * I_xx - I_xz * I_xy) * node_R_z[i]) / D
    node_lambda_z[i] = ((I_xz * I_yy - I_xy * I_yz) * node_R_x[i] + (I_yz * I_xx - I_xz * I_xy) * node_R_y[i] + (I_xy * I_xy - I_xx * I_yy) * node_R_z[i]) / D

def _create_normal_face_of_cell_2d(cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int[:,:]', face_normal: 'float[:,:]', cell_nf: 'float[:,:,:]'):

  # compute the outgoing normal faces for each cell

  for i in range(len(cell_faceid)):
    c_center = cell_center[i]

    for j in range(cell_faceid[i, -1]):
      fid = cell_faceid[i, j]
      f_center = face_center[fid]
      f_normal = face_normal[fid]

      snormal = c_center - f_center
      if (snormal[0] * f_normal[0] + snormal[1] * f_normal[1]) > 0.0:
        snormal *= -1

      cell_nf[i, j] = snormal

def _distance_2d(x: 'float[:]', y: 'float[:]'):
  return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def _dist_ortho_function_2d(d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]', face_dist_ortho: 'float[:]'):

  for i in range(d_boundaryfaces.shape[0]):
    bf = d_boundaryfaces[i]
    K = face_cellid[bf, 0]
    v = cell_center[K] - face_center[bf]
    u = face_normal[bf]  # /mesuref[i]
    projection = cell_center[K] - (v[0] * u[0] + v[1] * u[1]) * u
    face_dist_ortho[bf] = 2 * _distance_2d(cell_center[K].astype('float64'),
                                      projection.astype('float64'))  # +  distance_2d(ghostcenter[i], projection_bis)

  for i in range(d_innerfaces.shape[0]):
    bf = d_innerfaces[i]
    K = face_cellid[bf, 0]
    L = face_cellid[bf, 1]
    u = face_normal[bf]  # /mesuref[i]

    v = cell_center[K] - face_center[bf]
    projection = cell_center[K] - (v[0] * u[0] + v[1] * u[1]) * u

    v = cell_center[L] - face_center[bf]
    projection_bis = cell_center[L] - (v[0] * u[0] + v[1] * u[1]) * u
    face_dist_ortho[bf] = _distance_2d(cell_center[K].astype('float64'), projection.astype('float64')) \
                         + _distance_2d(cell_center[L].astype('float64'), projection_bis.astype('float64'))



# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################

def compile(func):
  #return func
  return numba.jit(nopython=True, fastmath=True, cache=True)(func)
def rcompile(func):
  return func

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
_distance_2d = compile(_distance_2d)
_search_halo_cell = compile(_search_halo_cell)

# public
append = compile(_append)
count_max_node_cellid = compile(_count_max_node_cellid)
create_node_cellid = compile(_create_node_cellid)
create_cellfid_and_bf_info = compile(_create_cellfid_and_bf_info)
count_max_cell_cellnid = compile(_count_max_cell_cellnid)
create_cell_cellnid = compile(_create_cell_cellnid)
create_info = compile(_create_info)
compute_cell_center_volume_2d = compile(_compute_cell_center_volume_2d)
compute_cell_center_volume_3d = compile(_compute_cell_center_volume_3d)
compute_face_info_2d = compile(_compute_face_info_2d)
compute_face_info_3d = compile(_compute_face_info_3d)
face_gradient_info_2d = compile(_face_gradient_info_2d)
face_gradient_info_3d = compile(_face_gradient_info_3d)
variables_2d = compile(_variables_2d)
variables_3d = compile(_variables_3d)
get_max_node_faceid = compile(_get_max_node_faceid)
get_node_faceid = compile(_get_node_faceid)
define_face_and_node_name = compile(_define_face_and_node_name)
append_1d = compile(_append_1d)
create_halo_cells = compile(_create_halo_cells)
create_ghost_info_2d = compile(_create_ghost_info_2d)
create_ghost_info_3d = compile(_create_ghost_info_3d)
get_ghost_part_size = compile(_get_bf_recv_part_info)
create_ghost_tables_2d = compile(_create_ghost_tables_2d)
create_ghost_tables_3d = compile(_create_ghost_tables_3d)
get_cell_ghostnid_size = compile(_get_cell_ghostnid_size)
create_cell_ghostnid = compile(_create_cell_ghostnid)
get_ghost_tables_size = compile(_get_ghost_tables_size)

count_max_bcell_halobfid = compile(_count_max_bcell_halobfid)
create_bcell_halobfid = compile(_create_bcell_halobfid)
create_ghost_new_index = compile(_create_ghost_new_index)
create_halo_ghost_tables_2d = compile(_create_halo_ghost_tables_2d)
create_halo_ghost_tables_3d = compile(_create_halo_ghost_tables_3d)

create_normal_face_of_cell_2d = compile(_create_normal_face_of_cell_2d)
dist_ortho_function_2d = compile(_dist_ortho_function_2d)

count_max_b_nodeid = compile(_count_max_b_nodeid)
create_b_nodeid = compile(_create_b_nodeid)
get_max_b_ncellid = compile(_get_max_b_ncellid)
create_b_ncellid = compile(_create_b_ncellid)