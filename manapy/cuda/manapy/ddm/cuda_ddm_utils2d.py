import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)


# manapy/ddm/ddm_utils2d.py

# ✅ ❌ 🔨
# create_cellsOfFace 🔨
# create_cell_faceid
# create_NeighborCellByFace
# create_node_cellid
# create_NormalFacesOfCell
# create_node_ghostid
# face_info_2d
# create_2d_halo_structure
# update_pediodic_info_2d
# Compute_2dcentervolumeOfCell
# create_2dfaces
# create_info_2dfaces
# face_gradient_info_2d
# variables_2d
# dist_ortho_function

    #_petype_fnmap = {
    #    'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
    #    'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
    #Create 2d faces


    #_petype_fnmap = {
    #    'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
    
    #    'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
    #                     [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
    
    #    'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
    #            'tri': [[0, 1, 2], [3, 4, 5]]},
    #    'pyr': {'quad': [[0, 1, 2, 3]],
    #            'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
    #}
    
def get_info(
  cells_nodes: 'int32[:, :]',
  nb_elements: 'int32',
):
  """
  cells_nodes: list of cells with their nodes id
  nb_elements: number of cells

  goal 
  - create neighbors by nodes and faces
  - create faces with their nodes
  - create cells with their faces
  """


  for i in range(nb_elements):
    cell = cells_nodes[i]
    pass
  

  


def get_kernel_create_cellsOfFace():
  
  def kernel_create_cellsOfFace(
    faceid:'int32[:,:]',
    nbelements:'int32',
    nbfaces:'int32',
    cellid:'int32[:,:]',
    maxcellfid:'int32'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, nbelements, stride):
      for j in range(faceid[i][-1]):
        if cellid[faceid[i][j]][0] == -1 :
          cellid[faceid[i][j]][0] = i

        if cellid[faceid[i][j]][0] != i:
          cellid[faceid[i][j]][0] = cellid[faceid[i][j]][0]
          cellid[faceid[i][j]][1] = i

  kernel_create_cellsOfFace = GPU_Backend.compile_kernel(kernel_create_cellsOfFace)
  
  def result(*args):
    VarClass.debug(kernel_create_cellsOfFace, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = args[1] #nbelements
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_create_cellsOfFace[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result




# ====================================================
# ====================================================
# ====================================================


def insertion_sort(arr : 'int[:]'):
  for i in range(1, len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and key < arr[j]:
      arr[j + 1] = arr[j]
      j -= 1
    arr[j + 1] = key


def is_in_array(array: 'int[:]', item : 'int') -> 'int':
  """
    check if item is in array
    return 1 if item in array otherwise 0

    Note:
      the number of item in the array is array[-1]
  """
  for i in range(array[-1]):
    if item == array[i]:
      return 1
  return 0


def intersect_nodes(face_nodes : 'int[:]', nb_nodes : 'int', neighbors_by_node : 'int[:, :]', intersect_cell : 'int[:]'):
  """
    Get the intersection cells of the face nodes

    Args:
      face_nodes: nodes of the face
      nb_nodes : number of nodes of the face
      neighbors_by_node: for each node get the neighbor cells

    Results:
      intersect_cell: common cells between all neighbors of each node (two at most)
  """
  index = 0

  intersect_cell[0] = -1
  intersect_cell[1] = -1
  
  cells = neighbors_by_node[face_nodes[0]]
  for i in range(cells[-1]):
    intersect_cell[index] = cells[i]
    for j in range(1, nb_nodes):
      if is_in_array(neighbors_by_node[face_nodes[j]], cells[i]) == 0:
        intersect_cell[index] = -1
        break
    if intersect_cell[index] != -1:
      index = index + 1
    if index >= 2:
      return


def create_faces(nodes : 'int[:]', out_faces: 'int[:, :]', size_info: 'int[:]', cell_type : 'int'):
  """
    create cell faces

    Args:
      nodes : nodes of the cell
      cell_type :
        # 4 => tetra
        # 8 => hexahedron
        # 5 => pyramid
    
    Return:
      out_faces: faces of the cell
      size_info: 
        contains number of nodes of each face
        and the total number of faces of the cell

    Used Map:
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


def has_the_same_items(arr1 : 'int[:]', arr2 : 'int[:]', size : 'int'):
  """
    check if all items of arr1 are in arr2
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

def has_face(cell_faces : 'int[:]', face : 'int[:]', face_size : 'int', faces : 'int[:, :]'):
  """
    - check `face` is in `cell_faces`
  """
  for i in range(cell_faces[-1]):
    face_nodes = faces[cell_faces[i]]
    if face_size == face_nodes[-1] and has_the_same_items(face_nodes, face):
      return cell_faces[i]
  return -1


def create_info(cells : 'int[:, :]', neighbors_by_node : 'int[:, :]'):
  """
    - Create faces
    - Create cells with their corresponding faces. 
    - Create cell neighbors of each face.
  """
  nb_cells = cells.shape[0]
  max_nb_faces = 6
  max_nb_nodes = 4
  nb_faces = nb_cells * max_nb_faces
  size_info = np.zeros(shape=(max_nb_faces + 1))
  cell_faces = np.zeros(shape=(max_nb_faces, max_nb_nodes))
  intersect_cells = np.zeros(2)
  faces = np.zeros(shape=(nb_faces, max_nb_nodes + 1))
  neighbors_by_face = np.zeros(shape=(nb_faces, 2))
  arr_cell_faces = np.zeros(shape=(nb_cells, max_nb_faces + 1))
  faces_counter = np.zeros(1)

  for i in range(cells.shape[0]):
    create_faces(cells[i], cell_faces, size_info, cells[i][-1])

    # For every face of cell[j]
    # Get the intersection of this face's nodes' neighboring cells.
    # The result should be two cells `intersect_cells`
    for j in range(size_info[-1]):
      intersect_nodes(cell_faces[j], size_info[j], neighbors_by_node, intersect_cells)

      # check if the face already created
      face_id = has_face(arr_cell_faces[i], cell_faces[j], size_info[j], faces)
      
      # Create face if note exist
      if face_id == -1:
        face_id = faces_counter[0]
        faces_counter[0] += 1
        # copy nodes from cell_faces
        for k in range(size_info[j]):
          faces[face_id][k] = cell_faces[k]
        faces[face_id][-1] = size_info[j]
      

      # Create cells with their corresponding faces.
      # The face has at most two neighbors
      # Assign this face to both of them
      tmp_size = arr_cell_faces[intersect_cells[0]][-1]
      arr_cell_faces[intersect_cells[0]][tmp_size] = face_id
      if intersect_cells[1] != -1:
        tmp_size = arr_cell_faces[intersect_cells[1]][-1]
        arr_cell_faces[intersect_cells[1]][tmp_size] = face_id
      
      # Create cell neighbors of each face
      # face neighbor cells
      neighbors_by_face[face_id][0] = intersect_cells[0]
      neighbors_by_face[face_id][1] = intersect_cells[1]
      
      

def create_neighbors_by_node(cells : 'int[:, :]', neighbors_by_node : 'int[:, :]'):
  """
    Create neighbor cells for each node

    Args:
      cells: cells with their nodes
    
    Return:
      neighbors_by_node
  """
  nb_cells = cells.shape[0]
  nodes_counter = np.zeros(shape=(1), dtype=np.int32)

  for i in range(nb_cells):
    for j in range(cells[i][-1]):
      node = neighbors_by_node[cells[i][j]]
      size = node[-1]
      node[-1] += 1
      node[size] = i
      if size == 0:
        nodes_counter[0] += 1



# Try with sort
# Try with size is the first item
# create neighbors of cell by  node