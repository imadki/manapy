import numpy as np
from manapy.domain import Mesh
from manapy.helpers import get_test_mesh
from manapy.testing.TestTables.aTestTables import ATestTables
from manapy.backends.config import ManapyConfig

class SmallCuboidTables(ATestTables):
  """
  A class serve as a test unit representing a cuboid split into 6 cuboids.

  CUBE VISUALIZATION:
  -------------------
  Dimensions: X=10, Y=5, Z=15

  (meshio representation)
  Cube Representation (8 nodes [0, 1, 2, 3, 4, 5, 6, 7]):
    y -> z -> x
    unit vector [1, 0.5, 1.5]
    Axes:
         ↑ Z
         |
         |
         •------→ X
        /
       /
      Y

       5-------4
      /|      /|  Face 1 [0, 1, 2, 3] 3
     1-------0 |  Face 2 [4, 7, 6, 5] 4
     | |     | |  Face 3 [0, 4, 7, 3] 1
     | 6-----|-7  Face 4 [1, 5, 6, 2] 2
     |/      |/   Face 5 [0, 4, 5, 1] 6
     2-------3    Face 6 [3, 7, 6, 2] 5
  """

  def __init__(self):
    super().__init__()
    mesh = Mesh(get_test_mesh("smallCuboid.msh")[1], 3, ManapyConfig.getDefaultConfig())
    self.meshio_cells = np.copy(mesh.cells)
    self.nb_cells = 8
    self.nb_faces = 36
    self.nb_ghosts = 24
    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)

    self.nodes = np.array([
      [0.0, 0.0, 0.0],
      [10.0, 0.0, 0.0],
      [10.0, 5.0, 0.0],
      [0.0, 5.0, 0.0],
      [0.0, 0.0, 15.0],
      [10.0, 0.0, 15.0],
      [10.0, 5.0, 15.0],
      [0.0, 5.0, 15.0],
      [5.0, 0.0, 0.0],
      [10.0, 2.5, 0.0],
      [5.0, 5.0, 0.0],
      [0.0, 2.5, 0.0],
      [5.0, 0.0, 15.0],
      [10.0, 2.5, 15.0],
      [5.0, 5.0, 15.0],
      [0.0, 2.5, 15.0],
      [10.0, 0.0, 7.5],
      [0.0, 0.0, 7.5],
      [10.0, 5.0, 7.5],
      [0.0, 5.0, 7.5],
      [5.0, 0.0, 7.5],
      [5.0, 5.0, 7.5],
      [0.0, 2.5, 7.5],
      [10.0, 2.5, 7.5],
      [5.0, 2.5, 15.0],
      [5.0, 2.5, 0.0],
      [5.0, 2.5, 7.5],
    ], dtype=np.float64)

    self.node_oldname = np.array([1, 1, 1, 1, 3, 2, 2, 3, 1, 1, 1, 1, 3, 2, 3, 3, 2, 4, 2, 4, 6, 5, 4, 2, 3, 1, 0], dtype=np.int32)

    # Array of cells (tetrahedrons).
    # Contains node connectivity for each cell.
    self.cells = np.array([
      [6, 14, 21, 18, 13, 24, 26, 23, 8],
      [13, 24, 26, 23, 5, 12, 20, 16, 8],
      [18, 21, 10, 2, 23, 26, 25, 9, 8],
      [23, 26, 25, 9, 16, 20, 8, 1, 8],
      [14, 7, 19, 21, 24, 15, 22, 26, 8],
      [24, 15, 22, 26, 12, 4, 17, 20, 8],
      [21, 19, 3, 10, 26, 22, 11, 25, 8],
      [26, 22, 11, 25, 20, 17, 0, 8, 8],
    ], dtype=np.int32)


    self.node_cellid = np.array([
      [7, 0, 0, 0, 0, 0, 0, 0, 1],
      [3, 0, 0, 0, 0, 0, 0, 0, 1],
      [2, 0, 0, 0, 0, 0, 0, 0, 1],
      [6, 0, 0, 0, 0, 0, 0, 0, 1],
      [5, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 1],
      [4, 0, 0, 0, 0, 0, 0, 0, 1],
      [3, 7, 0, 0, 0, 0, 0, 0, 2], # node 7 shows in cell 3 and 7
      [2, 3, 0, 0, 0, 0, 0, 0, 2],
      [2, 6, 0, 0, 0, 0, 0, 0, 2],
      [6, 7, 0, 0, 0, 0, 0, 0, 2],
      [1, 5, 0, 0, 0, 0, 0, 0, 2],
      [0, 1, 0, 0, 0, 0, 0, 0, 2],
      [0, 4, 0, 0, 0, 0, 0, 0, 2],
      [4, 5, 0, 0, 0, 0, 0, 0, 2],
      [1, 3, 0, 0, 0, 0, 0, 0, 2],
      [5, 7, 0, 0, 0, 0, 0, 0, 2],
      [0, 2, 0, 0, 0, 0, 0, 0, 2],
      [4, 6, 0, 0, 0, 0, 0, 0, 2],
      [1, 3, 5, 7, 0, 0, 0, 0, 4],
      [0, 2, 4, 6, 0, 0, 0, 0, 4],
      [4, 5, 6, 7, 0, 0, 0, 0, 4],
      [0, 1, 2, 3, 0, 0, 0, 0, 4],
      [0, 1, 4, 5, 0, 0, 0, 0, 4],
      [2, 3, 6, 7, 0, 0, 0, 0, 4],
      [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ], dtype=np.int32)

    # Indicates the neighboring cells by node for each cell.
    self.cell_cellnid = np.array([
      [1, 2, 3, 4, 5, 6, 7, 7], # combine node_cellid of cell_0 = [6, 14, 21, 18, 13, 24, 26, 23] Good luck doing that!!
      [0, 2, 3, 4, 5, 6, 7, 7],
      [0, 1, 3, 4, 5, 6, 7, 7],
      [0, 1, 2, 4, 5, 6, 7, 7],
      [0, 1, 2, 3, 5, 6, 7, 7],
      [0, 1, 2, 3, 4, 6, 7, 7],
      [0, 1, 2, 3, 4, 5, 7, 7],
      [0, 1, 2, 3, 4, 5, 6, 7],
    ], dtype=np.int32)

    # Faces array.
    # Node indices that form each face.
    # Constructing the hexahedron’s faces and removing any duplicates.
    self.faces = np.array([
      [6, 14, 21, 18, 4],
      [6, 14, 13, 24, 4],
      [14, 21, 24, 26, 4],
      [21, 18, 26, 23, 4],
      [6, 18, 13, 23, 4],
      [13, 24, 26, 23, 4],
      [13, 24, 5, 12, 4],
      [24, 26, 12, 20, 4],
      [26, 23, 20, 16, 4],
      [13, 23, 5, 16, 4],
      [5, 12, 20, 16, 4],
      [18, 21, 10, 2, 4],
      [21, 10, 26, 25, 4],
      [10, 2, 25, 9, 4],
      [18, 2, 23, 9, 4],
      [23, 26, 25, 9, 4],
      [26, 25, 20, 8, 4],
      [25, 9, 8, 1, 4],
      [23, 9, 16, 1, 4],
      [16, 20, 8, 1, 4],
      [14, 7, 19, 21, 4],
      [14, 7, 24, 15, 4],
      [7, 19, 15, 22, 4],
      [19, 21, 22, 26, 4],
      [24, 15, 22, 26, 4],
      [24, 15, 12, 4, 4],
      [15, 22, 4, 17, 4],
      [22, 26, 17, 20, 4],
      [12, 4, 17, 20, 4],
      [21, 19, 3, 10, 4],
      [19, 3, 22, 11, 4],
      [3, 10, 11, 25, 4],
      [26, 22, 11, 25, 4],
      [22, 11, 17, 0, 4],
      [11, 25, 0, 8, 4],
      [20, 17, 0, 8, 4],
    ], dtype=np.int32)

    # Lists the face IDs (referencing the 'faces' array) that compose each cell.
    self.cell_faceid = np.array([
      [0, 1, 2, 3, 4, 5, 6],
      [5, 6, 7, 8, 9, 10, 6],
      [11, 3, 12, 13, 14, 15, 6],
      [15, 8, 16, 17, 18, 19, 6],
      [20, 21, 22, 23, 2, 24, 6],
      [24, 25, 26, 27, 7, 28, 6],
      [29, 23, 30, 31, 12, 32, 6],
      [32, 27, 33, 34, 16, 35, 6],
    ], dtype=np.int32)

    self.face_cellid = np.array([
      [0, -1],
      [0, -1],
      [4, 0],
      [2, 0],
      [0, -1],
      [1, 0],
      [1, -1],
      [5, 1],
      [3, 1],
      [1, -1],
      [1, -1],
      [2, -1],
      [6, 2],
      [2, -1],
      [2, -1],
      [3, 2],
      [7, 3],
      [3, -1],
      [3, -1],
      [3, -1],
      [4, -1],
      [4, -1],
      [4, -1],
      [6, 4],
      [5, 4],
      [5, -1],
      [5, -1],
      [7, 5],
      [5, -1],
      [6, -1],
      [6, -1],
      [6, -1],
      [7, 6],
      [7, -1],
      [7, -1],
      [7, -1],
  ], dtype=np.int32)

    self.face_oldname = np.array([5, 3, 0, 0, 2, 0, 3, 0, 0, 2, 6, 5, 0, 1, 2, 0, 0, 1, 2, 6, 5, 3, 4, 0, 0, 3, 4, 0, 6, 5, 4, 1, 0, 4, 1, 6], dtype=np.int32)

    # Indicates the neighbor cell facing each of the cell's faces.
    self.cell_cellfid = np.array([
      [4, 2, 1, 0, 0, 0, 3],
      [0, 5, 3, 0, 0, 0, 3],
      [0, 6, 3, 0, 0, 0, 3],
      [2, 1, 7, 0, 0, 0, 3],
      [6, 0, 5, 0, 0, 0, 3],
      [4, 7, 1, 0, 0, 0, 3],
      [4, 2, 7, 0, 0, 0, 3],
      [6, 5, 3, 0, 0, 0, 3],
  ], dtype=np.int32)

    # Boundary faces, these are faces that all their nodes are in the same boundary surface.
    # There is six boundary faces in a cube.
    self.phy_faces = np.array([
      [4, 12, 17, 20, 4],
      [0, 8, 17, 20, 4],
      [5, 12, 16, 20, 4],
      [1, 8, 16, 20, 4],
      [6, 14, 18, 21, 4],
      [2, 10, 18, 21, 4],
      [7, 14, 19, 21, 4],
      [3, 10, 19, 21, 4],
      [7, 15, 19, 22, 4],
      [3, 11, 19, 22, 4],
      [4, 15, 17, 22, 4],
      [0, 11, 17, 22, 4],
      [5, 13, 16, 23, 4],
      [1, 9, 16, 23, 4],
      [6, 13, 18, 23, 4],
      [2, 9, 18, 23, 4],
      [4, 12, 15, 24, 4],
      [7, 14, 15, 24, 4],
      [5, 12, 13, 24, 4],
      [6, 13, 14, 24, 4],
      [0, 8, 11, 25, 4],
      [3, 10, 11, 25, 4],
      [1, 8, 9, 25, 4],
      [2, 9, 10, 25, 4],
    ], dtype=np.int32)

    # Map phyid to face_id
    self.phy_id_to_face_id = np.array([28, 35, 10, 19, 0, 11, 20, 29, 22, 30, 26, 33, 9, 18, 4, 14, 25, 21, 6, 1, 34, 31, 17, 13], dtype=np.int32)

    # Map face to (phyid/ghost/boundary face)
    self.face_to_phyid = np.array([4, 19, -1, -1, 14, -1, 18, -1, -1, 12, 2, 5, -1, 23, 15, -1, -1, 22, 13, 3, 6, 17, 8, -1, -1, 16, 10, -1, 0, 7, 9, 21, -1, 11, 20, 1], dtype=np.int32)

    # Ghost identifiers by phy_id per node.
    # Neighboring boundary faces for each node.
    self.node_ghostid = np.array([
      [1, 11, 20, 0, 3],
      [3, 13, 22, 0, 3],
      [5, 15, 23, 0, 3],
      [7, 9, 21, 0, 3],
      [0, 10, 16, 0, 3],
      [2, 12, 18, 0, 3],
      [4, 14, 19, 0, 3],
      [6, 8, 17, 0, 3],
      [1, 3, 20, 22, 4],
      [13, 15, 22, 23, 4],
      [5, 7, 21, 23, 4],
      [9, 11, 20, 21, 4],
      [0, 2, 16, 18, 4],
      [12, 14, 18, 19, 4],
      [4, 6, 17, 19, 4],
      [8, 10, 16, 17, 4],
      [2, 3, 12, 13, 4],
      [0, 1, 10, 11, 4],
      [4, 5, 14, 15, 4],
      [6, 7, 8, 9, 4],
      [0, 1, 2, 3, 4],
      [4, 5, 6, 7, 4],
      [8, 9, 10, 11, 4],
      [12, 13, 14, 15, 4],
      [16, 17, 18, 19, 4],
      [20, 21, 22, 23, 4],
      [0, 0, 0, 0, 0],
    ], dtype=np.int32)


    # Ghost identifiers by phy_id per cell.
    # Neighboring boundary faces for each cell.
    self.cell_ghostnid = np.array([
      [4, 14, 19, 6, 17, 5, 7, 15, 12, 18, 16, 13, 12],
      [12, 14, 18, 19, 16, 17, 13, 15, 2, 0, 1, 3, 12],
      [4, 5, 14, 15, 6, 7, 21, 23, 12, 13, 20, 22, 12],
      [12, 13, 14, 15, 20, 21, 22, 23, 2, 3, 0, 1, 12],
      [4, 6, 17, 19, 8, 7, 9, 5, 16, 18, 10, 11, 12],
      [16, 17, 18, 19, 8, 10, 9, 11, 0, 2, 1, 3, 12],
      [4, 5, 6, 7, 8, 9, 21, 23, 10, 11, 20, 22, 12],
      [8, 9, 10, 11, 20, 21, 22, 23, 0, 1, 2, 3, 12],
    ], dtype=np.int32)

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
      self.cell_center[i] = np.sum(vertices, axis=0) / 8.0
      self.cell_volume[i] = 93.75

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / 4.0

      v1 = vertices[1] - vertices[0]
      v2 = vertices[2] - vertices[0]
      normal = np.cross(v1, v2)

      self.face_center[i] = center
      self.face_normal[i] = normal
      self.face_measure[i] = np.linalg.norm(normal)

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










