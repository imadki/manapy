import numpy as np
from manapy.domain import Mesh
from manapy.tests.meshes import get_mesh

class SmallTriangleTables:
  """
  SmallTriangleTables is a 2D test fixture for a `10 x 5` rectangle subdivided
  into 4 smaller rectangles, each split into 2 triangles.

  MESH VISUALIZATION:
  -------------------
  Dimensions: X=10, Y=5

  Node layout:

    3-------6-------2
    |     / |     / |
    |   /   |   /   |
    | /     | /     |
    7-------8-------5
    |     / |     / |
    |   /   |   /   |
    | /     | /     |
    0-------4-------1
  """

  def __init__(self):
    mesh = Mesh(get_mesh("one_triangles.msh")[1], 2)
    self.meshio_cells = np.copy(mesh.cells)
    self.nb_cells = 8
    self.nb_faces = 16
    self.nb_ghosts = 8
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
      [5.0, 0.0, 0.0],
      [10.0, 2.5, 0.0],
      [5.0, 5.0, 0.0],
      [0.0, 2.5, 0.0],
      [5.0, 2.5, 0.0],
    ], dtype=np.float64)

    self.node_oldname = np.array([1, 2, 2, 1, 3, 2, 4, 1, 0], dtype=np.int32)

    # Array of cells (tetrahedrons).
    # Contains node connectivity for each cell.
    self.cells = np.array([
      [0, 4, 7, 3],
      [7, 4, 8, 3],
      [7, 8, 3, 3],
      [3, 8, 6, 3],
      [4, 1, 8, 3],
      [8, 1, 5, 3],
      [8, 5, 6, 3],
      [6, 5, 2, 3],
    ], dtype=np.int32)

    self.node_cellid = np.array([
      [0, 0, 0, 0, 0, 0, 1],
      [4, 5, 0, 0, 0, 0, 2], # node 1 shows in cell 4 and 5
      [7, 0, 0, 0, 0, 0, 1],
      [2, 3, 0, 0, 0, 0, 2],
      [0, 1, 4, 0, 0, 0, 3],
      [5, 6, 7, 0, 0, 0, 3],
      [3, 6, 7, 0, 0, 0, 3],
      [0, 1, 2, 0, 0, 0, 3],
      [1, 2, 3, 4, 5, 6, 6],
    ], dtype=np.int32)

    # Indicates the neighboring cells by node for each cell.
    self.cell_cellnid = np.array([
      [1, 2, 4, 0, 0, 0, 3], # combine node_cellid of cell_0 = [0, 4, 7, 3]
      [0, 2, 3, 4, 5, 6, 6],
      [0, 1, 3, 4, 5, 6, 6],
      [1, 2, 4, 5, 6, 7, 6],
      [0, 1, 2, 3, 5, 6, 6],
      [1, 2, 3, 4, 6, 7, 6],
      [1, 2, 3, 4, 5, 7, 6],
      [3, 5, 6, 0, 0, 0, 3],
    ], dtype=np.int32)

    # Faces array.
    # Node indices that form each face.
    # Constructing the hexahedron’s faces and removing any duplicates.
    self.faces = np.array([
      [0, 4, 2],
      [4, 7, 2],
      [7, 0, 2],
      [4, 8, 2],
      [8, 7, 2],
      [8, 3, 2],
      [3, 7, 2],
      [8, 6, 2],
      [6, 3, 2],
      [4, 1, 2],
      [1, 8, 2],
      [1, 5, 2],
      [5, 8, 2],
      [5, 6, 2],
      [5, 2, 2],
      [2, 6, 2],
    ], dtype=np.int32)

    # Lists the face IDs (referencing the 'faces' array) that compose each cell.
    self.cell_faceid = np.array([
      [0, 1, 2, 3],
      [1, 3, 4, 3],
      [4, 5, 6, 3],
      [5, 7, 8, 3],
      [9, 10, 3, 3],
      [10, 11, 12, 3],
      [12, 13, 7, 3],
      [13, 14, 15, 3],
    ], dtype=np.int32)

    self.face_cellid = np.array([
      [0, -1],
      [1, 0],
      [0, -1],
      [4, 1],
      [2, 1],
      [3, 2],
      [2, -1],
      [6, 3],
      [3, -1],
      [4, -1],
      [5, 4],
      [5, -1],
      [6, 5],
      [7, 6],
      [7, -1],
      [7, -1],
    ], dtype=np.int32)

    self.face_oldname = np.array([3, 0, 1, 0, 0, 0, 1, 0, 4, 3, 0, 2, 0, 0, 2, 4], dtype=np.int32)

    # Indicates the neighbor cell facing each of the cell's faces.
    self.cell_cellfid = np.array([
      [1, 0, 0, 1],
      [0, 4, 2, 3],
      [1, 3, 0, 2],
      [2, 6, 0, 2],
      [5, 1, 0, 2],
      [4, 6, 0, 2],
      [5, 7, 3, 3],
      [6, 0, 0, 1],
    ], dtype=np.int32)

    # Boundary faces, these are faces that all their nodes are in the same boundary surface.
    # There is six boundary faces in a cube.
    self.phy_faces = np.array([
      [0, 4, 2],
      [1, 4, 2],
      [1, 5, 2],
      [2, 5, 2],
      [2, 6, 2],
      [3, 6, 2],
      [3, 7, 2],
      [0, 7, 2],
    ], dtype=np.int32)

    # Map phyid to face_id
    self.phy_id_to_face_id = np.array([0, 9, 11, 14, 15, 8, 6, 2], dtype=np.int32)

    # Map face to (phyid/ghost/boundary face)
    self.face_to_phyid = np.array([0, -1, 7, -1, -1, -1, 6, -1, 5, 1, -1, 2, -1, -1, 3, 4], dtype=np.int32)

    # Ghost identifiers by phy_id per node.
    # Neighboring boundary faces for each node.
    self.node_ghostid = np.array([
      [0, 7, 2],
      [1, 2, 2],
      [3, 4, 2],
      [5, 6, 2],
      [0, 1, 2],
      [2, 3, 2],
      [4, 5, 2],
      [6, 7, 2],
      [0, 0, 0],
    ], dtype=np.int32)


    # Ghost identifiers by phy_id per cell.
    # Neighboring boundary faces for each cell.
    self.cell_ghostnid = np.array([
      [0, 7, 1, 6, 4],
      [6, 7, 0, 1, 4],
      [6, 7, 5, 0, 3],
      [5, 6, 4, 0, 3],
      [0, 1, 2, 0, 3],
      [1, 2, 3, 0, 3],
      [2, 3, 4, 5, 4],
      [4, 5, 2, 3, 4],
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
      self.cell_center[i] = np.sum(vertices, axis=0) / 3.0
      self.cell_volume[i] = 6.25

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / 2.0
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

def draw_rectangle_points_and_vertices():
  import matplotlib.pyplot as plt

  mesh = SmallTriangleTables()

  nodes = mesh.nodes
  cells = mesh.cells
  fig, ax = plt.subplots(figsize=(7, 4))

  xy = nodes[:, :2]
  ax.scatter(xy[:, 0], xy[:, 1], s=60)

  for i, (x, y) in enumerate(xy):
    ax.text(x + 0.12, y + 0.12, str(i), fontsize=10)

  for cell in cells:
    poly = np.vstack([xy[cell[:3]], xy[cell[0]]])
    ax.plot(poly[:, 0], poly[:, 1], linewidth=1)

  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("Rectangle points and vertices")
  ax.grid(True, alpha=0.3)
  plt.show()


if __name__ == "__main__":
  draw_rectangle_points_and_vertices()






