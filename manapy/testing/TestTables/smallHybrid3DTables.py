import numpy as np
from manapy.domain import Mesh
from manapy.helpers import get_test_mesh
from manapy.testing.TestTables.aTestTables import ATestTables
from manapy.backends.config import ManapyConfig


class SmallHybrid3DTables(ATestTables):
  r"""
  A class serving as a 3D test fixture for a hybrid mesh made of:
    - 10 tetrahedrons
    - 1 pyramid
    - 1 hexahedron

  The mesh represents two adjacent hexahedra along the x-axis. The left one is
  kept as a hexahedron, while the right one is decomposed into tetrahedra and a
  pyramid around the interior point `(1.4500005, 0.5, 0.5)`.
  """

  def __init__(self):
    super().__init__()
    mesh = Mesh(get_test_mesh("smallHybrid3d.msh")[1], 3, ManapyConfig.getDefaultConfig())
    self.meshio_cells = np.copy(mesh.cells)
    self.nb_cells = 12
    self.nb_faces = 33
    self.nb_ghosts = 15

    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)

    self.nodes = np.array([
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 1.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [1.0, 0.0, 1.0],
      [0.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [2.0, 1.0, 1.0],
      [2.0, 0.0, 1.0],
      [2.0, 1.0, 0.0],
      [1.4500005, 0.5, 0.5],
    ], dtype=np.float64)

    self.node_oldname = np.array([3, 3, 4, 1, 1, 1, 4, 1, 2, 2, 2, 2, 0], dtype=np.int32)

    # Array of cells (tetrahedrons, one pyramid, one hexahedron).
    self.cells = np.array([
      [6, 2, 12, 8, 0, 0, 0, 0, 4],
      [1, 8, 12, 2, 0, 0, 0, 0, 4],
      [9, 10, 12, 11, 0, 0, 0, 0, 4],
      [11, 10, 12, 8, 0, 0, 0, 0, 4],
      [9, 6, 12, 10, 0, 0, 0, 0, 4],
      [6, 9, 12, 0, 0, 0, 0, 0, 4],
      [10, 6, 12, 8, 0, 0, 0, 0, 4],
      [12, 1, 0, 11, 0, 0, 0, 0, 4],
      [9, 12, 0, 11, 0, 0, 0, 0, 4],
      [1, 8, 11, 12, 0, 0, 0, 0, 4],
      [2, 1, 0, 6, 12, 0, 0, 0, 5],
      [4, 1, 0, 3, 7, 2, 6, 5, 8],
    ], dtype=np.int32)

    self.node_cellid = np.array([
      [5, 7, 8, 10, 11, 0, 0, 0, 0, 0, 0, 5],
      [1, 7, 9, 10, 11, 0, 0, 0, 0, 0, 0, 5],
      [0, 1, 10, 11, 0, 0, 0, 0, 0, 0, 0, 4],
      [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 4, 5, 6, 10, 11, 0, 0, 0, 0, 0, 6],
      [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 1, 3, 6, 9, 0, 0, 0, 0, 0, 0, 5],
      [2, 4, 5, 8, 0, 0, 0, 0, 0, 0, 0, 4],
      [2, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 4],
      [2, 3, 7, 8, 9, 0, 0, 0, 0, 0, 0, 5],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ], dtype=np.int32)

    # Indicates the neighboring cells by node for each cell.
    self.cell_cellnid = np.array([
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
      [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],
      [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 0, 10],
      [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 0, 10],
      [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 11],
      [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 11],
      [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 11],
      [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 11],
      [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 11],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 11],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 11],
      [0, 1, 4, 5, 6, 7, 8, 9, 10, 0, 0, 9],
    ], dtype=np.int32)

    # Faces array.
    # Constructing each cell faces and removing the duplicate.
    self.faces = np.array([
      [6, 2, 12, 0, 3],
      [6, 2, 8, 0, 3],
      [6, 12, 8, 0, 3],
      [2, 12, 8, 0, 3],
      [1, 8, 12, 0, 3],
      [1, 8, 2, 0, 3],
      [1, 12, 2, 0, 3],
      [9, 10, 12, 0, 3],
      [9, 10, 11, 0, 3],
      [9, 12, 11, 0, 3],
      [10, 12, 11, 0, 3],
      [11, 10, 8, 0, 3],
      [11, 12, 8, 0, 3],
      [10, 12, 8, 0, 3],
      [9, 6, 12, 0, 3],
      [9, 6, 10, 0, 3],
      [6, 12, 10, 0, 3],
      [6, 9, 0, 0, 3],
      [6, 12, 0, 0, 3],
      [9, 12, 0, 0, 3],
      [10, 6, 8, 0, 3],
      [12, 1, 0, 0, 3],
      [12, 1, 11, 0, 3],
      [12, 0, 11, 0, 3],
      [1, 0, 11, 0, 3],
      [9, 0, 11, 0, 3],
      [1, 8, 11, 0, 3],
      [2, 1, 0, 6, 4],
      [4, 1, 0, 3, 4],
      [4, 1, 7, 2, 4],
      [0, 3, 6, 5, 4],
      [4, 3, 7, 5, 4],
      [7, 2, 6, 5, 4],
    ], dtype=np.int32)

    # Lists the face IDs (referencing the 'faces' array) that compose each cell.
    self.cell_faceid = np.array([
      [0, 1, 2, 3, 0, 0, 4],
      [4, 5, 6, 3, 0, 0, 4],
      [7, 8, 9, 10, 0, 0, 4],
      [10, 11, 12, 13, 0, 0, 4],
      [14, 15, 7, 16, 0, 0, 4],
      [14, 17, 18, 19, 0, 0, 4],
      [16, 20, 13, 2, 0, 0, 4],
      [21, 22, 23, 24, 0, 0, 4],
      [19, 9, 25, 23, 0, 0, 4],
      [26, 4, 22, 12, 0, 0, 4],
      [27, 6, 21, 18, 0, 0, 5],
      [28, 29, 27, 30, 31, 32, 6],
    ], dtype=np.int32)

    self.face_cellid = np.array([
      [10, 0],
      [0, -1],
      [6, 0],
      [1, 0],
      [9, 1],
      [1, -1],
      [10, 1],
      [4, 2],
      [2, -1],
      [8, 2],
      [3, 2],
      [3, -1],
      [9, 3],
      [6, 3],
      [5, 4],
      [4, -1],
      [6, 4],
      [5, -1],
      [10, 5],
      [8, 5],
      [6, -1],
      [10, 7],
      [9, 7],
      [8, 7],
      [7, -1],
      [8, -1],
      [9, -1],
      [11, 10],
      [11, -1],
      [11, -1],
      [11, -1],
      [11, -1],
      [11, -1],
    ], dtype=np.int32)

    self.face_oldname = np.array([0, 4, 0, 0, 0, 6, 0, 0, 2, 0, 0, 2, 0, 0, 0, 5, 0, 5, 0, 0, 4, 0, 0, 0, 3, 3, 6, 0, 3, 6, 5, 1, 4], dtype=np.int32)

    # Indicates the neighbor cell facing each of the cell's faces.
    self.cell_cellfid = np.array([
      [10, 6, 1, 0, 0, 0, 3],
      [9, 10, 0, 0, 0, 0, 3],
      [4, 8, 3, 0, 0, 0, 3],
      [2, 9, 6, 0, 0, 0, 3],
      [5, 2, 6, 0, 0, 0, 3],
      [4, 10, 8, 0, 0, 0, 3],
      [4, 3, 0, 0, 0, 0, 3],
      [10, 9, 8, 0, 0, 0, 3],
      [5, 2, 7, 0, 0, 0, 3],
      [1, 7, 3, 0, 0, 0, 3],
      [11, 1, 7, 5, 0, 0, 5],
      [10, 0, 0, 0, 0, 0, 1],
    ], dtype=np.int32)

    # Boundary faces, these are faces that all their nodes are in the same boundary surface.
    self.phy_faces = np.array([
      [3, 4, 5, 7, 4],
      [0, 1, 3, 4, 4],
      [1, 2, 4, 7, 4],
      [0, 3, 5, 6, 4],
      [2, 5, 6, 7, 4],
      [0, 6, 9, 0, 3],
      [6, 9, 10, 0, 3],
      [8, 10, 11, 0, 3],
      [9, 10, 11, 0, 3],
      [2, 6, 8, 0, 3],
      [6, 8, 10, 0, 3],
      [1, 2, 8, 0, 3],
      [1, 8, 11, 0, 3],
      [0, 1, 11, 0, 3],
      [0, 9, 11, 0, 3],
    ], dtype=np.int32)

    # Map phyid to face_id
    self.phy_id_to_face_id = np.array([31, 28, 29, 30, 32, 17, 15, 11, 8, 1, 20, 5, 26, 24, 25], dtype=np.int32)

    # Map face to (phyid/ghost/boundary face)
    self.face_to_phyid = np.array([
      -1, 9, -1, -1, -1, 11, -1, -1, 8, -1, -1,
      7, -1, -1, -1, 6, -1, 5, -1, -1, 10, -1,
      -1, -1, 13, 14, 12, -1, 1, 2, 3, 0, 4,
    ], dtype=np.int32)

    # Ghost identifiers by face_id per node.
    # Neighboring boundary faces for each node.
    self.node_ghostid = np.array([
      [1, 3, 5, 13, 14, 0, 5],
      [1, 2, 11, 12, 13, 0, 5],
      [2, 4, 9, 11, 0, 0, 4],
      [0, 1, 3, 0, 0, 0, 3],
      [0, 1, 2, 0, 0, 0, 3],
      [0, 3, 4, 0, 0, 0, 3],
      [3, 4, 5, 6, 9, 10, 6],
      [0, 2, 4, 0, 0, 0, 3],
      [7, 9, 10, 11, 12, 0, 5],
      [5, 6, 8, 14, 0, 0, 4],
      [6, 7, 8, 10, 0, 0, 4],
      [7, 8, 12, 13, 14, 0, 5],
      [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)

    # Ghost identifiers by face_id per cell.
    # Neighboring boundary faces for each cell.
    self.cell_ghostnid = np.array([
      [3, 4, 5, 6, 9, 10, 2, 11, 7, 12, 0, 0, 0, 10],
      [1, 2, 11, 12, 13, 7, 9, 10, 4, 0, 0, 0, 0, 9],
      [5, 6, 8, 14, 7, 10, 12, 13, 0, 0, 0, 0, 0, 8],
      [7, 8, 12, 13, 14, 6, 10, 9, 11, 0, 0, 0, 0, 9],
      [5, 6, 8, 14, 3, 4, 9, 10, 7, 0, 0, 0, 0, 9],
      [3, 4, 5, 6, 9, 10, 8, 14, 1, 13, 0, 0, 0, 10],
      [6, 7, 8, 10, 3, 4, 5, 9, 11, 12, 0, 0, 0, 10],
      [1, 2, 11, 12, 13, 3, 5, 14, 7, 8, 0, 0, 0, 10],
      [5, 6, 8, 14, 1, 3, 13, 7, 12, 0, 0, 0, 0, 9],
      [1, 2, 11, 12, 13, 7, 9, 10, 8, 14, 0, 0, 0, 10],
      [2, 4, 9, 11, 1, 12, 13, 3, 5, 14, 6, 10, 0, 12],
      [0, 1, 2, 11, 12, 13, 3, 5, 14, 4, 9, 6, 10, 13],
    ], dtype=np.int32)

    self.geometry()

  @staticmethod
  def _tetrahedron_volume(vertices):
    A = np.array(vertices[0])
    B = np.array(vertices[1])
    C = np.array(vertices[2])
    D = np.array(vertices[3])

    AB = B - A
    AC = C - A
    AD = D - A

    volume = abs(np.linalg.det([AB, AC, AD])) / 6.0
    return volume

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
      if self.cells[i, -1] == 8: # hexa
        self.cell_center[i] = np.sum(vertices, axis=0) / 8
        self.cell_volume[i] = 1.0
      elif self.cells[i, -1] == 5: # pyramid
        self.cell_center[i] = np.sum(vertices, axis=0) / 5
        self.cell_volume[i] = 0.15
      else: # tetra
        self.cell_center[i] = np.sum(vertices, axis=0) / 4
        self.cell_volume[i] = self._tetrahedron_volume(vertices)

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / self.faces[i, -1]

      v1 = vertices[1] - vertices[0]
      v2 = vertices[2] - vertices[0]
      normal = np.cross(v1, v2)

      self.face_center[i] = center
      if self.faces[i, -1] == 4: # square
        self.face_normal[i] = normal
        self.face_measure[i] = np.linalg.norm(normal)
      else: # triangle
        self.face_normal[i] = normal * 0.5
        self.face_measure[i] = np.linalg.norm(normal) * 0.5

    for phy_id in range(self.nb_ghosts):
      face_id = self.phy_id_to_face_id[phy_id]
      cell_id = self.face_cellid[face_id, 0]

      cell_center = self.cell_center[cell_id]
      face_center = self.face_center[face_id]
      face_normal = self.face_normal[face_id]
      face_oldname = self.face_oldname[face_id]
      face_index = np.where(self.cell_faceid[cell_id, 0:self.cell_faceid[cell_id, -1]] == face_id)[0][0]

      n_hat = face_normal / np.linalg.norm(face_normal)
      ghost_center = cell_center - 2 * np.dot(cell_center - face_center, n_hat) * n_hat

      self.ghost_info_flt[phy_id, 0:3] = ghost_center[:]
      self.ghost_info_flt[phy_id, 3] = 0.0
      self.ghost_info_flt[phy_id, 4:7] = face_center[:]
      self.ghost_info_flt[phy_id, 7:10] = face_normal[:]
      self.ghost_info_int[phy_id, 0] = cell_id
      self.ghost_info_int[phy_id, 1] = face_index
      self.ghost_info_int[phy_id, 2] = face_oldname
      self.ghost_info_int[phy_id, 3] = cell_id