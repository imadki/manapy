import numpy as np
from manapy.domain import Mesh
from manapy.helpers import get_test_mesh
from manapy.testing.TestTables.aTestTables import ATestTables

class CuboidTables(ATestTables):
  """
  A class serve as a test unit representing a cuboid split into cuboids.

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
    self.width = 10  # number of cubes along the x-axis, y-axis and z-axis
    self.WIDTH = 10.0
    self.HEIGHT = 5.0
    self.DEPTH = 15.0

    self.nb_cells = self.width * self.width * self.width
    self.nb_nodes = (self.width + 1) * (self.width + 1) * (self.width + 1)
    self.nb_faces = 3 * self.width * self.width * (self.width + 1)
    self.nb_ghosts = 6 * self.width * self.width

    self.x_length = self.WIDTH / self.width
    self.y_length = self.HEIGHT / self.width
    self.z_length = self.DEPTH / self.width

    mesh = Mesh(get_test_mesh("cuboid.msh")[1], 3)
    self.meshio_cells = np.copy(mesh.cells)

    self.cells = np.zeros((self.nb_cells, 9), dtype=np.int32)
    self.nodes = np.zeros((self.nb_nodes, 3), dtype=np.float64)
    self.faces = np.zeros((self.nb_faces, 5), dtype=np.int32)
    self.cell_faceid = np.zeros((self.nb_cells, 7), dtype=np.int32)
    self.phy_faces = np.zeros((self.nb_ghosts, 5), dtype=np.int32)
    self.face_to_phyid = np.ones(self.nb_faces, dtype=np.int32) * -1
    self.phy_id_to_face_id = np.zeros(self.nb_ghosts, dtype=np.int32)
    self.face_cellid = np.ones((self.nb_faces, 2), dtype=np.int32) * -1
    self.cell_cellnid = np.ones((self.nb_cells, 27), dtype=np.int32) * -1
    self.cell_cellfid = np.ones((self.nb_cells, 7), dtype=np.int32) * -1
    self.node_cellid = np.ones((self.nb_nodes, 9), dtype=np.int32) * -1
    self.cell_ghostnid = np.ones((self.nb_cells, 13), dtype=np.int32) * -1
    self.node_ghostid = np.ones((self.nb_nodes, 5), dtype=np.int32) * -1
    self.face_oldname = np.zeros(self.nb_faces, dtype=np.int32)
    self.node_oldname = np.zeros(self.nb_nodes, dtype=np.int32)
    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)

    self.node_cellid[:, -1] = 0
    self.cell_ghostnid[:, -1] = 0
    self.node_ghostid[:, -1] = 0
    self.cell_cellnid[:, -1] = 0

    self.create_mesh()
    self.geometry()

  def _node_id(self, ix, iy, iz):
    return ix * (self.width + 1) * (self.width + 1) + iz * (self.width + 1) + iy

  def _cell_id(self, ix, iy, iz):
    return ix * self.width * self.width + iz * self.width + iy

  def _y_face_id(self, ix, iy_plane, iz):
    return ix * self.width * (self.width + 1) + iz * (self.width + 1) + iy_plane

  def _x_face_id(self, ix_plane, iy, iz):
    offset = self.width * self.width * (self.width + 1)
    return offset + ix_plane * self.width * self.width + iz * self.width + iy

  def _z_face_id(self, ix, iy, iz_plane):
    offset = 2 * self.width * self.width * (self.width + 1)
    return offset + iz_plane * self.width * self.width + ix * self.width + iy

  def _append_value(self, table, row, value):
    index = table[row, -1]
    table[row, index] = value
    table[row, -1] += 1

  def _append_unique_value(self, table, row, value):
    size = table[row, -1]
    if size >= table.shape[1] - 1:
      return
    for k in range(size):
      if table[row, k] == value:
        return
    table[row, size] = value
    table[row, -1] += 1

  def _build_nodes(self):
    # The y-index increases from top to bottom to match the old helper ordering.
    node_index = 0
    for ix in range(self.width + 1):
      x = self.WIDTH - ix * self.x_length
      for iz in range(self.width + 1):
        z = self.DEPTH - iz * self.z_length
        for iy in range(self.width + 1):
          y = self.HEIGHT - iy * self.y_length
          self.nodes[node_index] = np.array([x, y, z], dtype=np.float64)
          node_index += 1

  def _build_cells(self):
    # Node order inside one cell:
    # [0,1,2,3] top loop, [4,5,6,7] bottom loop.
    for ix in range(self.width):
      for iz in range(self.width):
        for iy in range(self.width):
          cell_id = self._cell_id(ix, iy, iz)

          n0 = self._node_id(ix, iy, iz)
          n1 = self._node_id(ix + 1, iy, iz)
          n2 = self._node_id(ix + 1, iy, iz + 1)
          n3 = self._node_id(ix, iy, iz + 1)
          n4 = self._node_id(ix, iy + 1, iz)
          n5 = self._node_id(ix + 1, iy + 1, iz)
          n6 = self._node_id(ix + 1, iy + 1, iz + 1)
          n7 = self._node_id(ix, iy + 1, iz + 1)

          self.cells[cell_id] = np.array([n0, n1, n2, n3, n4, n5, n6, n7, 8], dtype=np.int32)

          # Face order matches the old helper:
          # 0=[0,1,2,3], 1=[4,7,6,5], 2=[0,4,7,3], 3=[1,5,6,2], 4=[0,4,5,1], 5=[3,7,6,2]
          self.cell_faceid[cell_id] = np.array([
            self._y_face_id(ix, iy, iz),
            self._y_face_id(ix, iy + 1, iz),
            self._x_face_id(ix, iy, iz),
            self._x_face_id(ix + 1, iy, iz),
            self._z_face_id(ix, iy, iz),
            self._z_face_id(ix, iy, iz + 1),
            6
          ], dtype=np.int32)

          self._append_value(self.node_cellid, n0, cell_id)
          self._append_value(self.node_cellid, n1, cell_id)
          self._append_value(self.node_cellid, n2, cell_id)
          self._append_value(self.node_cellid, n3, cell_id)
          self._append_value(self.node_cellid, n4, cell_id)
          self._append_value(self.node_cellid, n5, cell_id)
          self._append_value(self.node_cellid, n6, cell_id)
          self._append_value(self.node_cellid, n7, cell_id)

          self._set_cell_face_neighbors(cell_id)

  def _set_cell_face_neighbors(self, cell_id):
    # This follows the same neighbor order as the old hexa helper.
    size = self.width * self.width
    neighbors = np.array([
      cell_id - size,
      cell_id - self.width,
      cell_id - 1,
      cell_id + 1,
      cell_id + self.width,
      cell_id + size
    ], dtype=np.int32)

    if cell_id % self.width - 1 < 0:
      neighbors[2] = -1
    if cell_id % self.width + 1 >= self.width:
      neighbors[3] = -1
    if cell_id % size - self.width < 0:
      neighbors[1] = -1
    if cell_id % size + self.width >= size:
      neighbors[4] = -1
    if cell_id - size < 0:
      neighbors[0] = -1
    if cell_id + size >= self.nb_cells:
      neighbors[5] = -1

    neighbors = neighbors[neighbors != -1]
    self.cell_cellfid[cell_id, 0:len(neighbors)] = neighbors
    self.cell_cellfid[cell_id, -1] = len(neighbors)

  def _build_cell_node_neighbors(self):
    # Gather all distinct cells that share at least one node with the current cell.
    for cell_id in range(self.nb_cells):
      for node_id in self.cells[cell_id, 0:8]:
        node_count = self.node_cellid[node_id, -1]
        for j in range(node_count):
          neighbor = self.node_cellid[node_id, j]
          if neighbor != cell_id:
            self._append_unique_value(self.cell_cellnid, cell_id, neighbor)

  def _build_y_faces(self):
    # Faces orthogonal to y use the node order [0,1,2,3] from the cell description.
    face_id = 0
    for ix in range(self.width):
      for iz in range(self.width):
        for iy in range(self.width + 1):
          n0 = self._node_id(ix, iy, iz)
          n1 = self._node_id(ix + 1, iy, iz)
          n2 = self._node_id(ix + 1, iy, iz + 1)
          n3 = self._node_id(ix, iy, iz + 1)
          self.faces[face_id] = np.array([n0, n1, n2, n3, 4], dtype=np.int32)

          if iy == 0:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, 0, iz), -1], dtype=np.int32)
            self.face_oldname[face_id] = 3
          elif iy == self.width:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, self.width - 1, iz), -1], dtype=np.int32)
            self.face_oldname[face_id] = 4
          else:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, iy, iz), self._cell_id(ix, iy - 1, iz)], dtype=np.int32)
            self.face_oldname[face_id] = 0

          face_id += 1
    return face_id

  def _build_x_faces(self, face_id):
    # Faces orthogonal to x use the node order [0,4,7,3].
    for ix in range(self.width + 1):
      for iz in range(self.width):
        for iy in range(self.width):
          n0 = self._node_id(ix, iy, iz)
          n1 = self._node_id(ix, iy + 1, iz)
          n2 = self._node_id(ix, iy + 1, iz + 1)
          n3 = self._node_id(ix, iy, iz + 1)
          self.faces[face_id] = np.array([n0, n1, n2, n3, 4], dtype=np.int32)

          if ix == 0:
            self.face_cellid[face_id] = np.array([self._cell_id(0, iy, iz), -1], dtype=np.int32)
            self.face_oldname[face_id] = 1
          elif ix == self.width:
            self.face_cellid[face_id] = np.array([self._cell_id(self.width - 1, iy, iz), -1], dtype=np.int32)
            self.face_oldname[face_id] = 2
          else:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, iy, iz), self._cell_id(ix - 1, iy, iz)], dtype=np.int32)
            self.face_oldname[face_id] = 0

          face_id += 1
    return face_id

  def _build_z_faces(self, face_id):
    # Faces orthogonal to z use the node order [0,4,5,1].
    for iz in range(self.width + 1):
      for ix in range(self.width):
        for iy in range(self.width):
          n0 = self._node_id(ix, iy, iz)
          n1 = self._node_id(ix, iy + 1, iz)
          n2 = self._node_id(ix + 1, iy + 1, iz)
          n3 = self._node_id(ix + 1, iy, iz)
          self.faces[face_id] = np.array([n0, n1, n2, n3, 4], dtype=np.int32)

          if iz == 0:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, iy, 0), -1], dtype=np.int32)
            self.face_oldname[face_id] = 6
          elif iz == self.width:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, iy, self.width - 1), -1], dtype=np.int32)
            self.face_oldname[face_id] = 5
          else:
            self.face_cellid[face_id] = np.array([self._cell_id(ix, iy, iz), self._cell_id(ix, iy, iz - 1)], dtype=np.int32)
            self.face_oldname[face_id] = 0

          face_id += 1

  def _register_phy_face(self, phy_id, face_id):
    # Boundary nodes inherit the minimum oldname among all touching boundary faces.
    self.phy_faces[phy_id] = self.faces[face_id]
    self.face_to_phyid[face_id] = phy_id

    oldname = self.face_oldname[face_id]
    for node_id in self.faces[face_id, 0:4]:
      self._append_value(self.node_ghostid, node_id, phy_id)
      if self.node_oldname[node_id] == 0 or oldname < self.node_oldname[node_id]:
        self.node_oldname[node_id] = oldname

  def _build_boundary_faces(self):
    phy_id = 0

    for ix in range(self.width):
      for iz in range(self.width):
        self._register_phy_face(phy_id, self._y_face_id(ix, 0, iz))
        phy_id += 1

    for ix in range(self.width):
      for iz in range(self.width):
        self._register_phy_face(phy_id, self._y_face_id(ix, self.width, iz))
        phy_id += 1

    for iz in range(self.width):
      for iy in range(self.width):
        self._register_phy_face(phy_id, self._x_face_id(0, iy, iz))
        phy_id += 1

    for iz in range(self.width):
      for iy in range(self.width):
        self._register_phy_face(phy_id, self._x_face_id(self.width, iy, iz))
        phy_id += 1

    for ix in range(self.width):
      for iy in range(self.width):
        self._register_phy_face(phy_id, self._z_face_id(ix, iy, 0))
        phy_id += 1

    for ix in range(self.width):
      for iy in range(self.width):
        self._register_phy_face(phy_id, self._z_face_id(ix, iy, self.width))
        phy_id += 1

  def _build_cell_ghostnid(self):
    for cell_id in range(self.nb_cells):
      for node_id in self.cells[cell_id, 0:8]:
        node_ghost_count = self.node_ghostid[node_id, -1]
        for j in range(node_ghost_count):
          phy_id = self.node_ghostid[node_id, j]
          self._append_unique_value(self.cell_ghostnid, cell_id, phy_id)

  def _set_phy_id_to_face_id(self):
    mask = self.face_to_phyid != -1
    self.phy_id_to_face_id[self.face_to_phyid[mask]] = np.nonzero(mask)[0]

  def create_mesh(self):
    """
    Build the topological tables of the structured cube mesh.
    """
    self._build_nodes()
    self._build_cells()
    self._build_cell_node_neighbors()
    face_id = self._build_y_faces()
    face_id = self._build_x_faces(face_id)
    self._build_z_faces(face_id)
    self._build_boundary_faces()
    self._build_cell_ghostnid()
    self._set_phy_id_to_face_id()

  def geometry(self):
    """
    cell_center
    cell_volume
    face_center
    face_normal
    face_measure
    ghost_info_flt (0=ghostcenter_x&y&z, 3=gamma, 4=face_center_x&y&z, 7=face_normal_x&y&z)
    ghost_info_int (0=cell_id, 1=face index inside the cell, 2=face_oldname, 3=cell global id)
    """
    for cell_id in range(self.nb_cells):
      node_ids = self.cells[cell_id, 0:self.cells[cell_id, -1]]
      vertices = self.nodes[node_ids]
      self.cell_center[cell_id] = np.sum(vertices, axis=0) / 8.0
      self.cell_volume[cell_id] = self.x_length * self.y_length * self.z_length

    for face_id in range(self.nb_faces):
      node_ids = self.faces[face_id, 0:self.faces[face_id, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / 4.0

      v1 = vertices[1] - vertices[0]
      v2 = vertices[2] - vertices[0]
      normal = np.cross(v1, v2)

      self.face_center[face_id] = center
      self.face_normal[face_id] = normal
      self.face_measure[face_id] = np.linalg.norm(normal)


    for phy_id in range(self.nb_ghosts):
      face_id = self.phy_id_to_face_id[phy_id]
      cell_id = self.face_cellid[face_id, 0]
      face_index = np.where(self.cell_faceid[cell_id, 0:6] == face_id)[0][0]

      cell_center = self.cell_center[cell_id]
      face_center = self.face_center[face_id]
      face_normal = self.face_normal[face_id]
      n_hat = face_normal / np.linalg.norm(face_normal)
      ghost_center = cell_center - 2.0 * np.dot(cell_center - face_center, n_hat) * n_hat

      self.ghost_info_flt[phy_id, 0:3] = ghost_center
      self.ghost_info_flt[phy_id, 3] = 0.0
      self.ghost_info_flt[phy_id, 4:7] = face_center
      self.ghost_info_flt[phy_id, 7:10] = face_normal
      self.ghost_info_int[phy_id, 0] = cell_id
      self.ghost_info_int[phy_id, 1] = face_index
      self.ghost_info_int[phy_id, 2] = self.face_oldname[face_id]
      self.ghost_info_int[phy_id, 3] = cell_id
