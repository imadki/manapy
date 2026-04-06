import numpy as np
from manapy.domain import Mesh
from manapy.helpers import get_test_mesh
from manapy.testing.TestTables.aTestTables import ATestTables

class SmallTetrahedronTables(ATestTables):
  """
  A class serves as a test unit representing a cuboid split into 6 tetrahedrons.

  CUBE VISUALIZATION:
  -------------------
  The coordinates in 'self.nodes' form the following 3D object:
  Dimensions: X=10, Y=5, Z=15

        Z (15)
        ^
        |    7 *----------------* 6 (10,5,15)
        |     /|               /|
        |    / |              / |
 (0,0,15) 4 *----------------* 5| (10,0,15)
            |  |             |  |
            |  |             |  |
            |3 *-------------|--* 2 (10,5,0)
            | /              | /
            |/               |/
            *----------------*--------> X (10)
           0 (0,0,0)         1 (10,0,0)
          /
         / Y (5)

         0
        /|\        Face 1 [0, 1, 2]
       / | \       Face 2 [0, 1, 3]
      /  |  \      Face 3 [0, 2, 3]
     1---|---2     Face 4 [1, 2, 3]
      \  |  /
       \ | /
        \|/
         3

  (meshio representation)
  Cube Representation (8 nodes [0, 1, 2, 3, 4, 5, 6, 7]):
  The 6 Tetrahedrons (cells) are formed by the following node indices:
    - Tetra 1: [0, 1, 3, 4]
    - Tetra 2: [1, 3, 4, 5]
    - Tetra 3: [4, 5, 3, 7]
    - Tetra 4: [1, 3, 5, 2]
    - Tetra 5: [3, 7, 5, 2]
    - Tetra 6: [5, 7, 6, 2]

  Tetrahedron Representation:
  For a tetrahedron composed of 4 nodes [0, 1, 2, 3], its 4 faces are defined as:
    - Face 1: [0, 1, 2]
    - Face 2: [0, 1, 3]
    - Face 3: [0, 2, 3]
    - Face 4: [1, 2, 3]
  """

  def __init__(self):
    super().__init__()
    mesh = Mesh(get_test_mesh("smallTetrahedrons.msh")[1], 3)
    self.meshio_cells = np.copy(mesh.cells)
    self.nb_cells = 6
    self.nb_faces = 18
    self.nb_ghosts = 12
    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)

    self.nodes = np.array([
      [ 0.,  0.,  0.],
      [10.,  0.,  0.],
      [10.,  5.,  0.],
      [ 0.,  5.,  0.],
      [ 0.,  0., 15.],
      [10.,  0., 15.],
      [10.,  5., 15.],
      [ 0.,  5., 15.]
    ])

    self.node_oldname = np.array([
      1, 2, 2, 1, 1, 2, 2, 1
    ])

    # Array of cells (tetrahedrons).
    # Contains node connectivity for each cell.
    self.cells = np.array([
      [0, 1, 3, 4, 4],
      [1, 3, 4, 5, 4],
      [4, 5, 3, 7, 4],
      [1, 3, 5, 2, 4],
      [3, 7, 5, 2, 4],
      [5, 7, 6, 2, 4]
    ])

    self.node_cellid = np.array([
      [0, 0, 0, 0, 0, 1],
      [0, 1, 3, 0, 0, 3], # node 1 shows in cell 0, 1 and 3
      [3, 4, 5, 0, 0, 3],
      [0, 1, 2, 3, 4, 5],
      [0, 1, 2, 0, 0, 3],
      [1, 2, 3, 4, 5, 5],
      [5, 0, 0, 0, 0, 1],
      [2, 4, 5, 0, 0, 3]
    ])

    # Indicates the neighboring cells by node for each cell.
    self.cell_cellnid = np.array([
      [1, 2, 3, 4, 0, 4], # combine node_cellid of cell_0 = [0, 1, 3, 4]
      [0, 2, 3, 4, 5, 5],
      [0, 1, 3, 4, 5, 5],
      [0, 1, 2, 4, 5, 5],
      [0, 1, 2, 3, 5, 5],
      [1, 2, 3, 4, 0, 4]
    ])

    # Faces array.
    # Node indices that form each triangular face.
    # Constructing each tetrahedron faces and removing the duplicate.
    self.faces = np.array([
      [0, 1, 3, 3],
      [0, 1, 4, 3],
      [0, 3, 4, 3],
      [1, 3, 4, 3],
      [1, 3, 5, 3],
      [1, 4, 5, 3],
      [3, 4, 5, 3],
      [4, 5, 7, 3],
      [4, 3, 7, 3],
      [5, 3, 7, 3],
      [1, 3, 2, 3],
      [1, 5, 2, 3],
      [3, 5, 2, 3],
      [3, 7, 2, 3],
      [7, 5, 2, 3],
      [5, 7, 6, 3],
      [5, 6, 2, 3],
      [7, 6, 2, 3]
    ])

    # Lists the face IDs (referencing the 'faces' array) that compose each cell.
    self.cell_faceid = np.array([
      [0, 1, 2, 3, 4],
      [3, 4, 5, 6, 4],
      [6, 7, 8, 9, 4],
      [4, 10, 11, 12, 4],
      [9, 13, 12, 14, 4],
      [15, 14, 16, 17, 4]
    ])

    self.face_cellid = np.array([
      [0, -1],
      [0, -1],
      [0, -1],
      [1, 0],
      [3, 1],
      [1, -1],
      [2, 1],
      [2, -1],
      [2, -1],
      [4, 2],
      [3, -1],
      [3, -1],
      [4, 3],
      [4, -1],
      [5, 4],
      [5, -1],
      [5, -1],
      [5, -1]
    ])

    self.face_oldname = np.array([
      6, 4, 1, 0, 0, 4, 0, 5, 1, 0, 6, 2, 0, 3, 0, 5, 2, 3
    ])

    # Indicates the neighbor cell facing each of the cell's faces.
    self.cell_cellfid = np.array([
      [1, 0, 0, 0, 1],
      [0, 3, 2, 0, 3],
      [1, 4, 0, 0, 2],
      [1, 4, 0, 0, 2],
      [2, 3, 5, 0, 3],
      [4, 0, 0, 0, 1]
    ])

    # Boundary faces, these are faces that all their nodes are in the same boundary surface.
    # There is six boundary faces in a cube.
    self.phy_faces = np.array([
      [0, 1, 3, 3], # face_0
      [1, 2, 3, 3], # face_10
      [4, 5, 7, 3], # face_7
      [5, 6, 7, 3], # face_15
      [0, 1, 4, 3], # face_1
      [1, 4, 5, 3], # face_5
      [1, 2, 5, 3], # face_11
      [2, 5, 6, 3], # face_16
      [2, 3, 7, 3], # face_13
      [2, 6, 7, 3], # face_17
      [0, 3, 4, 3], # face_2
      [3, 4, 7, 3]  # face_8
    ])

    # Map phyid to face_id
    self.phy_id_to_face_id = np.array([ 0, 10,  7, 15,  1,  5, 11, 16, 13, 17,  2,  8])

    # Map face to (phyid/ghost/boundary face)
    self.face_to_phyid = np.array([0, 4, 10, -1, -1, 5, -1, 2, 11, -1, 1, 6, -1, 8, -1, 3, 7, 9])

    # Ghost identifiers by face_id per node.
    # Neighboring boundary faces for each node.
    self.node_ghostid = np.array([
      [0, 4, 10, 0, 0, 3],
      [0, 1, 4, 5, 6, 5],
      [1, 6, 7, 8, 9, 5],
      [0, 1, 8, 10, 11, 5],
      [2, 4, 5, 10, 11, 5],
      [2, 3, 5, 6, 7, 5],
      [3, 7, 9, 0, 0, 3],
      [2, 3, 8, 9, 11, 5]
    ])


    # Ghost identifiers by face_id per cell.
    # Neighboring boundary faces for each cell.
    self.cell_ghostnid = np.array([
      [0, 4, 10, 1, 5, 6, 8, 11, 2, 0, 0, 0, 9],
      [0, 1, 4, 5, 6, 8, 10, 11, 2, 3, 7, 0, 11],
      [2, 4, 5, 10, 11, 3, 6, 7, 0, 1, 8, 9, 12],
      [0, 1, 4, 5, 6, 8, 10, 11, 2, 3, 7, 9, 12],
      [0, 1, 8, 10, 11, 2, 3, 9, 5, 6, 7, 0, 11],
      [2, 3, 5, 6, 7, 8, 9, 11, 1, 0, 0, 0, 9]
    ])

    # Build the mesh geometry
    self.geometry()

  def geometry(self):
    """
    cell_center
    cell_volume
    face_center
    face_normal
    face_measure
    ghost_info_flt (0=ghostcenter_x&y&z, 3=gamma, 4=face_center_x&y&z, 7=face_normal_x&y&z)
    ghost_info_int # (0=cell_id, 1=face index inside the cell, 2=face_oldname, 3=cell global id)
    """
    for i in range(self.nb_cells):
      node_ids = self.cells[i, 0:self.cells[i, -1]]
      vertices = self.nodes[node_ids]
      self.cell_center[i] = np.sum(vertices, axis=0) / 4.0
      self.cell_volume[i] = 125.0

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / 3.0

      v1 = vertices[1] - vertices[0]
      v2 = vertices[2] - vertices[0]
      normal = np.cross(v1, v2)

      self.face_center[i] = center
      self.face_normal[i] = normal * 0.5 # TODO why
      self.face_measure[i] = np.linalg.norm(normal) * 0.5 # TODO why

    for phy_id in range(self.nb_ghosts):
      face_id = self.phy_id_to_face_id[phy_id]
      cell_id = self.face_cellid[face_id, 0]

      cell_center = self.cell_center[cell_id]
      face_center = self.face_center[face_id]
      face_normal = self.face_normal[face_id]
      face_oldname = self.face_oldname[face_id]
      face_index = np.where(self.cell_faceid[cell_id] == face_id)[0][0]

      n_hat = face_normal / np.linalg.norm(face_normal)
      ghost_center = cell_center - 2 * np.dot(cell_center - face_center, n_hat) * n_hat

      self.ghost_info_flt[phy_id, 0:3] = ghost_center[:]
      self.ghost_info_flt[phy_id, 3] = 0.0 #gamma
      self.ghost_info_flt[phy_id, 4:7] = face_center[:]
      self.ghost_info_flt[phy_id, 7:10] = face_normal[:]
      self.ghost_info_int[phy_id, 0] = cell_id
      self.ghost_info_int[phy_id, 1] = face_index
      self.ghost_info_int[phy_id, 2] = face_oldname
      self.ghost_info_int[phy_id, 3] = cell_id










