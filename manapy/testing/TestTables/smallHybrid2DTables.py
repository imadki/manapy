import numpy as np
from manapy.domain import Mesh
from manapy.helpers import get_test_mesh
from manapy.testing.TestTables.aTestTables import ATestTables
from manapy.backends.config import ManapyConfig


class SmallHybrid2DTables(ATestTables):
  r"""
  A class serve as a  2D test fixture for a `15 x 15` hybrid mesh made of
  10 triangles and 4 quadrilaterals.

  MESH VISUALIZATION:
  -------------------
    Dimensions: X=15, Y=15

  (3)------(8)------(11)------(2)
   |        |         |         |
   |        |         |         |
  (12)-----(5)------(6)------(13)
   |  \     |        | \       |
   |   \    |        |    \    |
  (14)-----(4)------(7)------(15)
   | \      | \       |  \    |
   |    \   |    \    |    \  |
  (0)------(9)------(10)----(1)
  """

  def __init__(self):
    super().__init__()
    mesh = Mesh(get_test_mesh("smallHybrid2D.msh")[1], 2, ManapyConfig.getDefaultConfig())
    self.meshio_cells = np.copy(mesh.cells)
    self.nb_cells = 14
    self.nb_faces = 29
    self.nb_ghosts = 12
    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)

    self.nodes = np.array([
      [0.0, 0.0, 0.0],
      [15.0, 0.0, 0.0],
      [15.0, 15.0, 0.0],
      [0.0, 15.0, 0.0],
      [5.0, 5.0, 0.0],
      [5.0, 10.0, 0.0],
      [10.0, 10.0, 0.0],
      [10.0, 5.0, 0.0],
      [5.0, 15.0, 0.0],
      [5.0, 0.0, 0.0],
      [10.0, 0.0, 0.0],
      [10.0, 15.0, 0.0],
      [0.0, 10.0, 0.0],
      [15.0, 10.0, 0.0],
      [0.0, 5.0, 0.0],
      [15.0, 5.0, 0.0],
    ], dtype=np.float64)

    self.node_oldname = np.array([3, 2, 1, 1, 0, 0, 0, 0, 1, 3, 3, 1, 4, 2, 4, 2], dtype=np.int32)

    # Array of cells (triangles and quadrilaterals).
    # Contains node connectivity for each cell.
    self.cells = np.array([
      [13, 6, 15, 0, 3],
      [15, 6, 7, 0, 3],
      [5, 12, 4, 0, 3],
      [4, 12, 14, 0, 3],
      [15, 7, 1, 0, 3],
      [1, 7, 10, 0, 3],
      [7, 4, 10, 0, 3],
      [10, 4, 9, 0, 3],
      [4, 14, 9, 0, 3],
      [9, 14, 0, 0, 3],
      [3, 8, 5, 12, 4],
      [8, 11, 6, 5, 4],
      [11, 2, 13, 6, 4],
      [6, 5, 4, 7, 4],
    ], dtype=np.int32)

    self.node_cellid = np.array([
      [9, 0, 0, 0, 0, 0, 1],
      [4, 5, 0, 0, 0, 0, 2],
      [12, 0, 0, 0, 0, 0, 1],
      [10, 0, 0, 0, 0, 0, 1],
      [2, 3, 6, 7, 8, 13, 6],
      [2, 10, 11, 13, 0, 0, 4],
      [0, 1, 11, 12, 13, 0, 5],
      [1, 4, 5, 6, 13, 0, 5],
      [10, 11, 0, 0, 0, 0, 2],
      [7, 8, 9, 0, 0, 0, 3],
      [5, 6, 7, 0, 0, 0, 3],
      [11, 12, 0, 0, 0, 0, 2],
      [2, 3, 10, 0, 0, 0, 3],
      [0, 12, 0, 0, 0, 0, 2],
      [3, 8, 9, 0, 0, 0, 3],
      [0, 1, 4, 0, 0, 0, 3],
    ], dtype=np.int32)

    self.cell_cellnid = np.array([
      [1, 4, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 5],
      [0, 4, 5, 6, 11, 12, 13, 0, 0, 0, 0, 0, 7],
      [3, 6, 7, 8, 10, 11, 13, 0, 0, 0, 0, 0, 7],
      [2, 6, 7, 8, 9, 10, 13, 0, 0, 0, 0, 0, 7],
      [0, 1, 5, 6, 13, 0, 0, 0, 0, 0, 0, 0, 5],
      [1, 4, 6, 7, 13, 0, 0, 0, 0, 0, 0, 0, 5],
      [1, 2, 3, 4, 5, 7, 8, 13, 0, 0, 0, 0, 8],
      [2, 3, 5, 6, 8, 9, 13, 0, 0, 0, 0, 0, 7],
      [2, 3, 6, 7, 9, 13, 0, 0, 0, 0, 0, 0, 6],
      [3, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
      [2, 3, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 4],
      [0, 1, 2, 10, 12, 13, 0, 0, 0, 0, 0, 0, 6],
      [0, 1, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 4],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 12],
    ], dtype=np.int32)

    # Faces array.
    # Node indices that form each line face.
    # Contains the unique edges extracted from the hybrid cells.
    self.faces = np.array([
      [13, 6, 2],
      [6, 15, 2],
      [15, 13, 2],
      [6, 7, 2],
      [7, 15, 2],
      [5, 12, 2],
      [12, 4, 2],
      [4, 5, 2],
      [12, 14, 2],
      [14, 4, 2],
      [7, 1, 2],
      [1, 15, 2],
      [7, 10, 2],
      [10, 1, 2],
      [7, 4, 2],
      [4, 10, 2],
      [4, 9, 2],
      [9, 10, 2],
      [14, 9, 2],
      [14, 0, 2],
      [0, 9, 2],
      [3, 8, 2],
      [8, 5, 2],
      [12, 3, 2],
      [8, 11, 2],
      [11, 6, 2],
      [6, 5, 2],
      [11, 2, 2],
      [2, 13, 2],
    ], dtype=np.int32)

    # Lists the face IDs (referencing the 'faces' array) that compose each cell.
    self.cell_faceid = np.array([
      [0, 1, 2, 0, 3],
      [1, 3, 4, 0, 3],
      [5, 6, 7, 0, 3],
      [6, 8, 9, 0, 3],
      [4, 10, 11, 0, 3],
      [10, 12, 13, 0, 3],
      [14, 15, 12, 0, 3],
      [15, 16, 17, 0, 3],
      [9, 18, 16, 0, 3],
      [18, 19, 20, 0, 3],
      [21, 22, 5, 23, 4],
      [24, 25, 26, 22, 4],
      [27, 28, 0, 25, 4],
      [26, 7, 14, 3, 4],
    ], dtype=np.int32)

    self.face_cellid = np.array([
      [12, 0],
      [1, 0],
      [0, -1],
      [13, 1],
      [4, 1],
      [10, 2],
      [3, 2],
      [13, 2],
      [3, -1],
      [8, 3],
      [5, 4],
      [4, -1],
      [6, 5],
      [5, -1],
      [13, 6],
      [7, 6],
      [8, 7],
      [7, -1],
      [9, 8],
      [9, -1],
      [9, -1],
      [10, -1],
      [11, 10],
      [10, -1],
      [11, -1],
      [12, 11],
      [13, 11],
      [12, -1],
      [12, -1],
    ], dtype=np.int32)

    self.face_oldname = np.array([0, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 3, 0, 0, 0, 3, 0, 4, 3, 1, 0, 4, 1, 0, 0, 1, 2], dtype=np.int32)

    self.cell_cellfid = np.array([
      [12, 1, 0, 0, 2],
      [0, 13, 4, 0, 3],
      [10, 3, 13, 0, 3],
      [2, 8, 0, 0, 2],
      [1, 5, 0, 0, 2],
      [4, 6, 0, 0, 2],
      [13, 7, 5, 0, 3],
      [6, 8, 0, 0, 2],
      [3, 9, 7, 0, 3],
      [8, 0, 0, 0, 1],
      [11, 2, 0, 0, 2],
      [12, 13, 10, 0, 3],
      [0, 11, 0, 0, 2],
      [11, 2, 6, 1, 4],
    ], dtype=np.int32)

    # Boundary faces, these are faces whose nodes lie on the same boundary edge.
    # There are twelve boundary faces in this hybrid mesh.
    self.phy_faces = np.array([
      [3, 8, 2],
      [3, 12, 2],
      [8, 11, 2],
      [2, 11, 2],
      [2, 13, 2],
      [13, 15, 2],
      [12, 14, 2],
      [1, 15, 2],
      [1, 10, 2],
      [9, 10, 2],
      [0, 9, 2],
      [0, 14, 2],
    ], dtype=np.int32)

    self.phy_id_to_face_id = np.array([21, 23, 24, 27, 28, 2, 8, 11, 13, 17, 20, 19], dtype=np.int32)

    # Map face to (phyid/ghost/boundary face)
    self.face_to_phyid = np.array([-1, -1, 5, -1, -1, -1, -1, -1, 6, -1, -1, 7, -1, 8, -1, -1, -1, 9, -1, 11, 10, 0, -1, 1, 2, -1, -1, 3, 4], dtype=np.int32)

    # Ghost identifiers by phy_id per node.
    # Neighboring boundary faces for each node.
    self.node_ghostid = np.array([
      [10, 11, 2],
      [7, 8, 2],
      [3, 4, 2],
      [0, 1, 2],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 2, 2],
      [9, 10, 2],
      [8, 9, 2],
      [2, 3, 2],
      [1, 6, 2],
      [4, 5, 2],
      [6, 11, 2],
      [5, 7, 2],
    ], dtype=np.int32)

    # Ghost identifiers by phy_id per cell.
    # Neighboring boundary faces for each cell.
    self.cell_ghostnid = np.array([
      [4, 5, 7, 0, 3],
      [5, 7, 0, 0, 2],
      [1, 6, 0, 0, 2],
      [1, 6, 11, 0, 3],
      [5, 7, 8, 0, 3],
      [7, 8, 9, 0, 3],
      [8, 9, 0, 0, 2],
      [8, 9, 10, 0, 3],
      [6, 11, 9, 10, 4],
      [9, 10, 6, 11, 4],
      [0, 1, 2, 6, 4],
      [0, 2, 3, 0, 3],
      [2, 3, 4, 5, 4],
      [0, 0, 0, 0, 0],
    ], dtype=np.int32)

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
      if self.cells[i, -1] == 3:
        self.cell_center[i] = np.sum(vertices, axis=0) / 3
        self.cell_volume[i] = 12.5
      else:
        self.cell_center[i] = np.sum(vertices, axis=0) / 4
        self.cell_volume[i] = 25

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / self.faces[i, -1]
      tangent = vertices[1] - vertices[0]
      normal = np.array([-tangent[1], tangent[0], 0.0], dtype=np.float64)

      self.face_center[i] = center
      self.face_normal[i] = normal
      self.face_measure[i] = np.sqrt(np.dot(tangent, tangent))

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


def draw_rectangle_points_and_vertices():
  import matplotlib.pyplot as plt

  mesh = SmallHybrid2DTables()

  nodes = mesh.nodes
  cells = mesh.cells
  fig, ax = plt.subplots(figsize=(7, 7))

  xy = nodes[:, :2]
  ax.scatter(xy[:, 0], xy[:, 1], s=60)

  for i, (x, y) in enumerate(xy):
    ax.text(x + 0.15, y + 0.15, str(i), fontsize=10)

  for cell in cells:
    nb_nodes = cell[-1]
    poly = np.vstack([xy[cell[:nb_nodes]], xy[cell[0]]])
    ax.plot(poly[:, 0], poly[:, 1], linewidth=1)

  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("Hybrid mesh points and vertices")
  ax.grid(True, alpha=0.3)
  plt.show()


if __name__ == "__main__":
  draw_rectangle_points_and_vertices()
