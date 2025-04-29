import numpy as np
import meshio
import numba
import time

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


def _append(cells, cells_item: 'int[:]', counter: 'int'):
  for i in range(len(cells_item)):
    cells[counter, 0:len(cells_item[i])] = cells_item[i]
    cells[counter, -1] = len(cells_item[i])
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
  cell_cellnid: 'int[:, :]',
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
        if nc != i and (size <= 0 or cell_cellnid[nc, size - 1] != i):
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
      if binary_search(node_cellid[face_nodes[j]], cells[i]) == -1:
        intersect_cell[index] = -1
        break
    if intersect_cell[index] != -1:
      index = index + 1
    if index >= 2:
      return


def _create_cell_faces_n(cells: 'int[:, :]', tmp_cell_faces: 'int[:, :, :]', tmp_size_info: 'int[:, :]', cell_type_map: 'int[:]'):
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

  for i in range(cells.shape[0]):
    nodes = cells[i]
    out_faces = tmp_cell_faces[i]
    size_info = tmp_size_info[i]
    cell_type = cell_type_map[i]

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

# @numba.jit(nopython=True)
# def _create_info(
#   cells: 'int[:, :]',
#   node_cellid: 'int[:, :]',
#   cell_type: 'int[:]',
#   tmp_cell_faces: 'int[:, :, :]',
#   tmp_size_info: 'int[:, :]',
#   tmp_cell_faces_map: 'int[:, :]',
#   faces: 'int[:, :]',
#   cell_faceid: 'int[:, :]',
#   face_cellid: 'int[:, :]',
#   cell_cellfid: 'int[:, :]',
#   faces_counter: 'int[:]'
# ):
#   """
#     - Create faces
#     - Create cells with their corresponding faces (cells.cellfid).
#     - Create neighboring cells for each face (faces.cellid).
#     - Create neighboring cells of a cell by face (cells.cellid).
#
#     Args:
#       cells: cells with their nodes (cell => cell nodes)
#       node_cellid: neighbor cells of each node (node => neighbor cells)
#       max_nb_nodes : maximum number of nodes on faces
#       max_nb_faces : maximum number of faces on cells
#
#     Return:
#       faces : (face => face nodes)
#       cell_faces : (cell => cell faces)
#       face_cellid : (face => neighboring cells of the face)
#       faces_counter : array(1) face counter
#       cell_cellfid : (cell => neighboring cells of a cell by face)
#
#   """
#   intersect_cells = np.zeros(2, dtype=np.int32)
#   _create_cell_faces_n(cells, tmp_cell_faces, tmp_size_info, cell_type)
#   nb_faces = (tmp_cell_faces_map.shape[1] - 1) // 2
#
#   for i in range(cells.shape[0]):
#     # For every face of the cell[i]
#     # Get the intersection of the neighboring cells of this face's nodes (N*n*n)
#     # The result should be two cells `intersect_cells`
#     for j in range(tmp_size_info[i, -1]):
#       _intersect_nodes(tmp_cell_faces[i, j], tmp_size_info[i, j], node_cellid, intersect_cells)
#       # The face has at most two neighbors
#       # swap to make intersect_cells[0] = cell_i id
#       if intersect_cells[1] == i:
#         intersect_cells[1] = intersect_cells[0]
#         intersect_cells[0] = i
#
#       face_id = -1
#       # Check if the face already exist
#       if intersect_cells[1] != -1:
#         for k in range(tmp_cell_faces_map[i, -1]):
#           if tmp_cell_faces_map[i, k] == intersect_cells[1]:
#             face_id = tmp_cell_faces_map[i, nb_faces + k]
#
#       if face_id == -1:
#         face_id = faces_counter[0]
#         faces_counter[0] += 1
#         # copy nodes from tmp_cell_faces
#         for k in range(tmp_size_info[i, j]):
#           faces[face_id, k] = tmp_cell_faces[i, j, k]
#         faces[face_id, -1] = tmp_size_info[i, j]
#
#         # Store the face in tmp_cell_faces_map for later existence verification.
#         if intersect_cells[1] != -1:
#           a = tmp_cell_faces_map[intersect_cells[1]]
#           size = a[-1]
#           a[size] = i
#           a[nb_faces + size] = face_id
#           a[-1] += 1
#
#       # (cell_faces) Create cell faces
#       cell_faceid[i, j] = face_id
#       cell_faceid[i, -1] += 1
#
#       # (face_cellid) Create neighboring cells of each face
#       face_cellid[face_id, 0] = intersect_cells[0]
#       face_cellid[face_id, 1] = intersect_cells[1]
#
#       # (cell_cellfid) Create neighboring cells of the cell by face
#       if intersect_cells[1] != -1:
#         cell_cellfid[i, j] = intersect_cells[1]
#         cell_cellfid[i, -1] += 1

# @numba.jit(nopython=True)

def _create_info(
  cells: 'int[:, :]',
  node_cellid: 'int[:, :]',
  cell_type: 'int[:]',
  tmp_cell_faces: 'int[:, :, :]',
  tmp_size_info: 'int[:, :]',
  tmp_cell_faces_map: 'int[:, :]',
  faces: 'int[:, :]',
  cell_faceid: 'int[:, :]',
  face_cellid: 'int[:, :]',
  cell_cellfid: 'int[:, :]',
  faces_counter: 'int[:]'
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

def compile(func):
  # return func
  return numba.jit(nopython=True, fastmath=True, cache=True)(func)

_is_in_array = compile(_is_in_array)
binary_search = compile(binary_search)
_append = compile(_append)
_count_max_node_cellid = compile(_count_max_node_cellid)
_create_node_cellid = compile(_create_node_cellid)
_count_max_cell_cellnid = compile(_count_max_cell_cellnid)
_create_cell_cellnid = compile(_create_cell_cellnid)
_intersect_nodes = compile(_intersect_nodes)
_create_cell_faces = compile(_create_cell_faces)
_create_info = compile(_create_info)

class Domain:
  def __init__(self, mesh_path, dim):
    meshio_mesh_dic, meshio_mesh_points = self._read_mesh(mesh_path)

    if not (isinstance(dim, int) and dim == 2 or dim == 3):
      raise ValueError('Invalid dimension')
    self.dim = dim

    start = time.time()
    print("Init")
    (
      self.max_cell_faceid,
      self.max_cell_nodeid,
      self.max_face_nodeid,
      self.nb_cells
    ) = self._init(meshio_mesh_dic, self.dim)
    print("_create_cells")
    (
      self.cells,
      self.cell_type
    ) = self._create_cells(meshio_mesh_dic, self.nb_cells, self.max_cell_nodeid, self.dim)

    self.nodes = meshio_mesh_points
    self.nb_nodes = np.int32(len(meshio_mesh_points))
    self.nb_cells = np.int32(len(self.cells))
    print("node_cellid")
    self.max_node_cellid = self._count_max_node_cellid(self.cells, self.nb_nodes)
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes, self.max_node_cellid)
    print("cell_cellnid")
    self.max_cell_cellnid = self._count_max_cell_cellnid(self.cells, self.node_cellid)
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid, self.max_cell_cellnid)
    print("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid,
      self.faces_counter
    ) = self._create_info(self.cells, self.node_cellid, self.cell_type, self.max_cell_faceid, self.max_face_nodeid)
    end = time.time()
    print(f"Execution time: {end - start:.6f} seconds")

  def _read_mesh(self, mesh_path):
    meshio_mesh = meshio.read(mesh_path)
    MESHIO_VERSION = int(meshio.__version__.split(".")[0])
    if MESHIO_VERSION < 4:
      meshio_mesh_dic = meshio_mesh.cells
    else:
      meshio_mesh_dic = meshio_mesh.cells_dict

    return meshio_mesh_dic, meshio_mesh.points

  def _init(self, meshio_mesh_dic, dim):
    max_cell_faceid = -1
    max_cell_nodeid = -1
    max_face_nodeid = -1
    for item in meshio_mesh_dic.keys():
      print(item)
      if item == 'triangle':
        max_cell_faceid = max(max_cell_faceid, 3)
        max_face_nodeid = max(max_face_nodeid, 2)
        max_cell_nodeid = max(max_cell_nodeid, 3)
      elif item == 'quad':
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 2)
        max_cell_nodeid = max(max_cell_nodeid, 4)
      elif item == 'tetra':
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 3)
        max_cell_nodeid = max(max_cell_nodeid, 4)
      elif item == 'hexahedron':
        max_cell_faceid = max(max_cell_faceid, 6)
        max_face_nodeid = max(max_face_nodeid, 4)
        max_cell_nodeid = max(max_cell_nodeid, 8)
      elif item == 'pyramid':
        max_cell_faceid = max(max_cell_faceid, 5)
        max_face_nodeid = max(max_face_nodeid, 4)
        max_cell_nodeid = max(max_cell_nodeid, 5)


    number_of_cells = 0
    for item in meshio_mesh_dic.keys():
      if dim == 3 and (item == 'triangle' or item == 'quad'):
        continue
      number_of_cells += len(meshio_mesh_dic[item])

    return (
      max_cell_faceid,
      max_cell_nodeid,
      max_face_nodeid,
      number_of_cells
    )

  def _create_cells(self, meshio_mesh_dic, number_of_cells, max_cell_nodeid, dim):
    cell_type_dic = {
      "triangle": 1,
      "quad": 2,
      "tetra": 3,
      "hexahedron": 4,
      "pyramid": 5,
    }

    cells = np.zeros(shape=(number_of_cells, max_cell_nodeid + 1), dtype=np.int32)
    cell_type = np.zeros(shape=(number_of_cells), dtype=np.int32)

    counter = 0
    for item in meshio_mesh_dic.keys():
      if dim == 3 and (item == 'triangle' or item == 'quad'):
        continue
      cells_item = meshio_mesh_dic[item]
      cell_type[counter:counter + len(cells_item)] = cell_type_dic[item]
      _append(cells, cells_item, counter)
      counter += len(cells_item)

    return (cells, cell_type)

  # ###############################
  # ###############################

  def _count_max_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'np.int32'):
    res = np.zeros(shape=(nb_nodes), dtype=np.int32)
    _count_max_node_cellid(cells, res)
    return np.max(res)

  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int', max_node_cellid: 'np.int32'):
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32)
    _create_node_cellid(cells, node_cellid)
    return node_cellid

  def _count_max_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    visited = np.zeros(cells.shape[0], dtype=np.int32)
    return _count_max_cell_cellnid(cells, node_cellid, visited)

  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]', max_cell_cellnid: 'int'):
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=np.int32)
    _create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  def _create_info(
    self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int',
  ):
    # ? Create tables
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(nb_cells, max_cell_faceid, max_face_nodeid), dtype=np.int32)
    # tmp_size_info = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)
    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.ones(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32) * -1
    faces[:, -1] = 0
    cell_faceid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_faceid[:, -1] = 0
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_cellfid[:, -1] = 0
    faces_counter = np.zeros(shape=(1), dtype=np.int32)
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
    # ? Return
    faces = faces[:faces_counter[0]]
    face_cellid = face_cellid[:faces_counter[0]]

    return (
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid,
      faces_counter[0]
    )