import numpy as np
from numba import cuda
from numba import jit


#@jit(nopython=True)
def insertion_sort(arr : 'int[:]'):
  for i in range(1, len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and key < arr[j]:
      arr[j + 1] = arr[j]
      j -= 1
    arr[j + 1] = key


#@jit(nopython=True)
def _is_in_array(array: 'int[:]', item : 'int') -> 'int':
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


#@jit(nopython=True)
def _intersect_nodes(face_nodes : 'int[:]', nb_nodes : 'int', node_cell_neighbors : 'int[:, :]', intersect_cell : 'int[:]'):
  """
    Get the common cells of neighboring cells of the face's nodes.

    Details:
    Identify the neighboring cells associated with each of the nodes that belong to a specific face.
    After identifying the neighboring cells for each of these nodes, we are interested in finding the common cells that are shared among all these neighboring cells.

    Args:
      face_nodes: nodes of the face
      nb_nodes : number of nodes of the face
      node_cell_neighbors: for each node get the neighbor cells

    Return:
      intersect_cell: array(2) common cells between all neighbors of each node (two at most)
  """
  index = 0

  intersect_cell[0] = -1
  intersect_cell[1] = -1
  
  cells = node_cell_neighbors[face_nodes[0]]
  for i in range(cells[-1]):
    intersect_cell[index] = cells[i]
    for j in range(1, nb_nodes):
      if _is_in_array(node_cell_neighbors[face_nodes[j]], cells[i]) == 0:
        intersect_cell[index] = -1
        break
    if intersect_cell[index] != -1:
      index = index + 1
    if index >= 2:
      return


#@jit(nopython=True)
def _create_faces(nodes : 'int[:]', out_faces: 'int[:, :]', size_info: 'int[:]', cell_type : 'int'):
  """
    Create cell faces

    Args:
      nodes : nodes of the cell
      cell_type :
        4 => tetra
        8 => hexahedron
        5 => pyramid
    
    Return:
      out_faces: faces of the cell
      size_info: 
        size_info[:-1] contains number of nodes of each face
        size_info[-1] total number of faces of the cell

    Notes:
    Used Map (h5py):
    'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},

    'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                     [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},

    'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
            'tri': [[0, 1, 2], [3, 4, 5]]},

    'pyr': {'quad': [[0, 1, 2, 3]],
            'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}

  """


  if cell_type == 4:
    out_faces[0][0] = nodes[0]
    out_faces[0][1] = nodes[1]
    out_faces[0][2] = nodes[2]
    size_info[0] = 3 #number of nodes

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
    
    size_info[-1] = 4 #number of faces

  if cell_type == 8:
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
  
  if cell_type == 5:
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


#@jit(nopython=True)
def _has_the_same_items(arr1 : 'int[:]', arr2 : 'int[:]', size : 'int'):
  """
    Check if all items of arr1 are in arr2

    Note:
      Not fast for large arrays
  """
  for i in range(size):
    found = 0
    for j in range(size):
      if arr1[j] == arr2[i]:
        found = 1
        break
    if found == 0:
      return 0
  return 1


#@jit(nopython=True)
def _has_face(cell_faces : 'int[:]', face : 'int[:]', face_size : 'int', faces : 'int[:, :]'):
  """
    Check if `face` is in `cell_faces`

    Args:
      cell_faces: cell => cell's face ids
      face: face's node ids
      face_size: number of nodes of the face.
      faces: face_id => nodes of the face.
    
    Return:
      The face id if it exist in `cell_faces` otherwise -1

  """
  for i in range(cell_faces[-1]):
    face_nodes = faces[cell_faces[i]]
    if face_size == face_nodes[-1] and _has_the_same_items(face_nodes, face, face_size):
      return cell_faces[i]
  return -1


#@jit(nopython=True)
def create_info(
  cells : 'int[:, :]',
  node_cell_neighbors : 'int[:, :]',
  faces : 'int[:, :]',
  cell_faces : 'int[:, :]',
  face_cell_neighbors : 'int[:, :]',
  cells_neighbors_by_face : 'int[:, :]',
  faces_counter : 'int[:]',
  max_nb_nodes : 'int', 
  max_nb_faces : 'int'
  ):
  """
    - Create faces
    - Create cells with their corresponding faces. 
    - Create neighboring cells for each face.
    - Create neighboring cells of cell by face.

    Args:
      cells: cells with their nodes (cell => cell nodes)
      node_cell_neighbors: neighbor cells of each node (node => neighbor cells)
      max_nb_nodes : maximum number of nodes on faces
      max_nb_faces : maximum number of faces on cells
    
    Return:
      faces : (face => face nodes)
      cell_faces : (cell => cell faces)
      face_cell_neighbors : (face => neighboring cells of the face)
      faces_counter : array(1) face counter
      cells_neighbors_by_face : (cell => neighboring cells of the cell by face)

  """

  tmp_cell_faces = np.zeros(shape=(max_nb_faces, max_nb_nodes), dtype=np.int32)
  tmp_size_info = np.zeros(shape=(max_nb_faces + 1), dtype=np.int32)
  intersect_cells = np.zeros(2, dtype=np.int32)

  for i in range(cells.shape[0]):
    _create_faces(cells[i], tmp_cell_faces, tmp_size_info, cells[i][-1])

    # For every face of the cell[i]
    # Get the intersection of the neighboring cells of this face's nodes
    # The result should be two cells `intersect_cells`
    for j in range(tmp_size_info[-1]):
      _intersect_nodes(tmp_cell_faces[j], tmp_size_info[j], node_cell_neighbors, intersect_cells)

      # check if the face already created
      face_id = _has_face(cell_faces[i], tmp_cell_faces[j], tmp_size_info[j], faces)
      
      # Create face if not exist
      if face_id == -1:
        face_id = faces_counter[0]
        faces_counter[0] += 1
        # copy nodes from tmp_cell_faces
        for k in range(tmp_size_info[j]):
          faces[face_id][k] = tmp_cell_faces[j][k]
        faces[face_id][-1] = tmp_size_info[j]
      

      # Create cells with their corresponding faces.
      # The face has at most two neighbors
      # Assign the face to both of them
      the_cell = cell_faces[intersect_cells[0]]
      if _is_in_array(the_cell, face_id) == 0:
        the_cell[the_cell[-1]] = face_id
        the_cell[-1] += 1
      if intersect_cells[1] != -1 and _is_in_array(cell_faces[intersect_cells[1]], face_id) == 0:
        the_cell = cell_faces[intersect_cells[1]]
        the_cell[the_cell[-1]] = face_id
        the_cell[-1] += 1
      
      # Create neighboring cells of each face
      face_cell_neighbors[face_id][0] = intersect_cells[0]
      face_cell_neighbors[face_id][1] = intersect_cells[1]

      # Create neighboring cells of the cell by face

      # print(cells_neighbors_by_face[i])
      # print(i)
      tmp = cells_neighbors_by_face[i]
      # print(i, "=>", tmp)
      if intersect_cells[0] == i and intersect_cells[1] != -1:
        tmp[tmp[-1]] = intersect_cells[1]
        tmp[-1] += 1
      elif intersect_cells[1] == i and intersect_cells[0] != -1:
        tmp[tmp[-1]] = intersect_cells[0]
        tmp[-1] += 1

def count_max_neighboring_cells_of_node(cells : 'int[:, :]', nb_nodes: 'int'):
  """
    Determine the max neighboring cells of a node
  """
  res = np.zeros(shape=(nb_nodes), dtype=np.int32)
  for cell in cells:
    for i in range(cell[-1]):
      node = cell[i]
      res[node] += 1
  return np.max(res)    


def create_node_neighboring_cells(cells : 'int[:, :]', node_cell_neighbors : 'int[:, :]'):
  """
    Create neighboring cells for each node

    Args:
      cells: cells with their nodes
    
    Return:
      node_cell_neighbors
  """

  for i in range(cells.shape[0]):
    for j in range(cells[i][-1]):
      node = node_cell_neighbors[cells[i][j]]
      size = node[-1]
      node[-1] += 1
      node[size] = i



def get_max_neighboring_cells_per_cell_nodes(
  cells : 'int[:, :]',
  node_neighbors : 'int[:, :]',
):
  """
    Get the maximum number of neighboring cells per cell's nodes across the mesh
  
    Details:
    For each cell in the mesh, we need to examine its nodes and count the cells that neighbor those nodes.
    to get all neighboring cells of the cell
    Then, determine the highest number of neighboring cells

    Args:
      cells: (cell_id => nodes of the cell)
      node_neighbors: (node_id => neighboring cells of the node)
    
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
      node_n = node_neighbors[cells[i][j]]
      for k in range(node_n[-1]):
        if node_n[k] != i and visited[node_n[k]] != i:
          visited[node_n[k]] = i
          counter += 1
    max_counter = max(max_counter, counter)
  return max_counter



def create_cell_neighbors_by_node(
    cells : 'int[:, :]',
    node_neighbors : 'int[:, :]',
    cells_neighbors_by_node : 'int[:, :]',
  ):
  """
    Get all neighboring cells by collecting adjacent cells from each node of the cell.

    Args:
      cells: (cell_id => nodes of the cell)
      node_neighbors: neighboring cells of each node
    
    Return:
      cells_neighbors_by_node: neighboring cells through the cell's nodes.
    
  """
  for i in range(cells.shape[0]):
    for j in range(cells[i][-1]):
      node_n = node_neighbors[cells[i][j]]
      for k in range(node_n[-1]):
        if node_n[k] != i and _is_in_array(cells_neighbors_by_node[i], node_n[k]) == 0:
          size = cells_neighbors_by_node[i][-1]
          cells_neighbors_by_node[i][-1] += 1
          cells_neighbors_by_node[i][size] = node_n[k]




# a debug function to create n cube
#@jit(nopython=True)
def create_cells(nb_x : 'int'):
  cells = np.zeros(shape=(nb_x, 9), dtype=np.int32)
  cells[0][0] = 0
  cells[0][1] = 1
  cells[0][2] = 2
  cells[0][3] = 3
  cells[0][-1] = 8
  for i in range(nb_x):
    cells[i][4] = cells[i][0] + 4
    cells[i][5] = cells[i][1] + 4
    cells[i][6] = cells[i][2] + 4
    cells[i][7] = cells[i][3] + 4

    if i + 1 < nb_x:
      cells[i + 1][0] = cells[i][4]
      cells[i + 1][1] = cells[i][5]
      cells[i + 1][2] = cells[i][6]
      cells[i + 1][3] = cells[i][7]
    cells[i][-1] = 8
  return cells


#######################################
#######################################
#############   Start   ###############      
#######################################
#######################################

###################################
# Read mesh data
###################################

def init():
  from manapy.base.base import Struct
  from manapy.ddm import Domain
  from manapy.partitions import MeshPartition

  dim = 3
  #mesh_path = "/home/aben-ham/Desktop/work/stage/manapy/mesh/3D/cube_bis.msh"
  mesh_path = "/home/aben-ham/Desktop/work/stage/manapy/mesh/3D/cube.msh"

  # dim = 2
  #mesh_path = "/home/aben-ham/Desktop/work/stage/manapy/mesh/2D/carre.msh"

  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="single")
  MeshPartition(mesh_path, dim=dim, conf=running_conf, periodic=[0,0,0])

  running_conf = Struct(backend="numba", signature=True, cache =True, float_precision="single")
  domain = Domain(dim=dim, conf=running_conf)
  
  return (domain)

domain = init()


# The only information known after reading the mesh `read_mesh_file`
nodeid = domain._cells._nodeid
vertex = domain._nodes._vertex




##################################################
# Create tables and get the essential information
##################################################

cells = nodeid
nb_cells = cells.shape[0]
nb_nodes = len(vertex)
max_nb_faces = 6
max_nb_nodes = 4
max_node_neighbor_cells = count_max_neighboring_cells_of_node(cells, nb_nodes)
nb_faces = nb_cells * max_nb_faces

# tables 

cell_faces = np.zeros(shape=(nb_cells, max_nb_faces + 1), dtype=np.int32)
face_cell_neighbors = np.zeros(shape=(nb_faces, 2), dtype=np.int32)

faces = np.zeros(shape=(nb_faces, max_nb_nodes + 1), dtype=np.int32)
faces_counter = np.zeros(1, dtype=np.int32)

node_cell_neighbors = np.zeros(shape=(nb_nodes, max_node_neighbor_cells + 1), dtype=np.int32)



# ============================



# create node cell neighbor
create_node_neighboring_cells(cells, node_cell_neighbors)

# create cell neighbor cells by node
max_nb_cell_neighbors_by_node = get_max_neighboring_cells_per_cell_nodes(cells, node_cell_neighbors)
cells_neighbors_by_node = np.zeros(shape=(nb_cells, max_nb_cell_neighbors_by_node + 1), dtype=np.int32)
cells_neighbors_by_face = np.zeros(shape=(nb_cells, max_nb_faces + 1), dtype=np.int32)
create_cell_neighbors_by_node(cells, node_cell_neighbors, cells_neighbors_by_node)

# create faces - cell faces - neighboring cells of the cell by face - neighboring cells of the face
create_info(cells, node_cell_neighbors, faces, cell_faces, face_cell_neighbors, cells_neighbors_by_face, faces_counter, max_nb_nodes, max_nb_faces)
face_cell_neighbors = face_cell_neighbors[0:faces_counter[0]]
faces = faces[0:faces_counter[0]]

#######################################
# Checkers
#######################################

def check_cells(my_cells, my_faces, cells, faces):
  print("domain cells shape: ", cells.shape)
  print("domain faces shape: ", faces.shape)
  print("test cells shape: ", my_cells.shape)
  print("test faces shape: ", my_faces.shape)
  for i in range(my_cells.shape[0]):
    my_cell = my_cells[i]
    cell = cells[i]
    if my_cell[-1] != cell[-1]:
      raise RuntimeError(f"Not the same {my_cell} {cell}")
    for j in range(my_cell[-1]):
      my_face = my_faces[my_cell[j]].sort()
      face = faces[cell[j]].sort()
      if np.array_equal(my_face, face) == False:
        raise RuntimeError(f"Not the same as will {face} {my_face}")
  print("cell faces: Ok!")
  print("face nodes: Ok!")
  print("")


def check_neighboring_by_face(my_nbr_by_face, nbr_by_face):
  print(my_nbr_by_face.shape)
  print(nbr_by_face.shape)

  if my_nbr_by_face.shape[0] != nbr_by_face.shape[0]:
    raise RuntimeError("Error: neighboring cells by face")
  for i in range(my_nbr_by_face.shape[0]):
    a = my_nbr_by_face[i][0:my_nbr_by_face[i][-1]]
    a.sort()
    b = nbr_by_face[i][0:nbr_by_face[i][-1]]
    b.sort()
    if np.array_equal(a, b) == False:
      print(a, b)
      raise RuntimeError("Error: neighboring cells by face")
  print("cell Neighboring cells by face: Ok!")
  print("")

# ==================================
# ==================================





print("\n Test File Result #########################\n")

print("Number of cells : ", len(cells))
print("Number of nodes : ", len(vertex))


print("number of faces: ", faces_counter[0])
print("node neighboring cells shape: ", node_cell_neighbors.shape)
print("face neighboring cells shape: ", face_cell_neighbors.shape)
print("number of faces that has one neighbor cell: ", np.count_nonzero(face_cell_neighbors[:, 1] == -1))
print("cell neighbors by node shape: ", cells_neighbors_by_node.shape)

######################################
print(" Domain #########################\n")
######################################

print("Number of cells : ", len(domain._cells._cellnid))
print("Number of nodes : ", domain._nbnodes)

print("number of faces: ", domain._faces._nodeid.shape[0])
print("node neighboring cells shape: ", domain._nodes.cellid.shape)
print("face neighboring cells shape: ", domain._faces._cellid.shape)
print("number of faces that has one neighbor cell: ", np.count_nonzero(domain._faces._cellid[:, 1] == -1))
print("cell neighbors by node shape: ", domain._cells._cellnid.shape)


######################################
print(" Check cell and faces data #########################\n")
######################################

# check is domain cell are the same as computed cells
# check is domain faces are the same as computed faces
check_cells(cell_faces, faces, domain._cells._faceid, domain._faces._nodeid)


######################################
print(" Check cells neighboring by face #########################\n")
######################################
check_neighboring_by_face(cells_neighbors_by_face, domain._cells._cellfid)
# Fix
# tmp_cell_faces
# tmp_size_info

# create neighbors of cell by  node

# Try local
# Try with sort


