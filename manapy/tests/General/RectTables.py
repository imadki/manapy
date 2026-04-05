import numpy as np
from manapy.domain import Mesh
from manapy.tests.meshes import get_mesh

class RectTables:
  def __init__(self):
    self.width = 10  # number of rectangles along the x-axis and y-axis
    self.WIDTH = 10.0  # mesh width
    self.HEIGHT = 5.0  # mesh height

    self.nb_cells = self.width * self.width
    self.nb_nodes = (self.width + 1) * (self.width + 1)
    self.nb_faces = 2 * self.width * (self.width + 1)
    self.nb_ghosts = 4 * self.width

    self.x_length = self.WIDTH / self.width
    self.y_length = self.HEIGHT / self.width

    mesh = Mesh(get_mesh("rectangles.msh")[1], 2)
    self.meshio_cells = mesh.cells
    self.cells = np.zeros((self.nb_cells, 5), dtype=np.int32)
    self.nodes = np.zeros((self.nb_nodes, 3), dtype=np.float64)
    self.faces = np.zeros((self.nb_faces, 3), dtype=np.int32)
    self.cell_faceid = np.zeros((self.nb_cells, 5), dtype=np.int32)
    self.phy_faces = np.zeros((self.nb_ghosts, 3), dtype=np.int32)
    self.face_to_phyid = np.ones(self.nb_faces, dtype=np.int32) * -1
    self.phy_id_to_face_id = np.zeros(self.nb_ghosts, dtype=np.int32)
    self.face_cellid = np.ones((self.nb_faces, 2), dtype=np.int32) * -1
    self.cell_cellnid = np.ones((self.nb_cells, 9), dtype=np.int32) * -1
    self.cell_cellfid = np.ones((self.nb_cells, 5), dtype=np.int32) * -1
    self.node_cellid = np.ones((self.nb_nodes, 5), dtype=np.int32) * -1
    self.cell_ghostnid = np.ones((self.nb_cells, 5), dtype=np.int32) * -1
    self.node_ghostid = np.ones((self.nb_nodes, 3), dtype=np.int32) * -1
    self.face_oldname = np.zeros(self.nb_faces, dtype=np.int32)
    self.node_oldname = np.zeros(self.nb_nodes, dtype=np.int32)
    self.cell_center = np.zeros((self.nb_cells, 3), dtype=np.float64)
    self.cell_volume = np.zeros(self.nb_cells, dtype=np.float64)
    self.face_center = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_normal = np.zeros((self.nb_faces, 3), dtype=np.float64)
    self.face_measure = np.zeros(self.nb_faces, dtype=np.float64)
    self.ghost_info_flt = np.zeros((self.nb_ghosts, 10), dtype=np.float64)
    self.ghost_info_int = np.zeros((self.nb_ghosts, 4), dtype=np.int32)
    self.cell_nf = np.zeros((self.nb_cells, 4, 3), dtype=np.float64)
    self.node_cellid[:, -1] = 0
    self.cell_ghostnid[:, -1] = 0
    self.node_ghostid[:, -1] = 0

    self.create_mesh()
    self.geometry()

  def _node_id(self, ix, iy):
    return ix * (self.width + 1) + iy

  def _cell_id(self, ix, iy):
    return ix * self.width + iy

  def _horizontal_face_id(self, ix, iy):
    return ix * (self.width + 1) + iy

  def _vertical_face_id(self, ix, iy):
    horizontal_faces = self.width * (self.width + 1)
    return horizontal_faces + ix * self.width + iy

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
    # Structured nodes are stored row by row on the logical grid.
    node_index = 0
    for ix in range(self.width + 1):
      x = ix * self.x_length
      for iy in range(self.width + 1):
        y = iy * self.y_length
        self.nodes[node_index] = np.array([x, y, 0.0], dtype=np.float64)
        node_index += 1

  def _build_cells(self):
    # Quad cell node order is [bottom-left, bottom-right, top-right, top-left].
    cell_index = 0
    for ix in range(self.width):
      for iy in range(self.width):
        n0 = self._node_id(ix, iy)
        n1 = self._node_id(ix + 1, iy)
        n2 = self._node_id(ix + 1, iy + 1)
        n3 = self._node_id(ix, iy + 1)

        self.cells[cell_index] = np.array([n0, n1, n2, n3, 4], dtype=np.int32)

        # Local face order is [bottom, right, top, left].
        f0 = self._horizontal_face_id(ix, iy)
        f1 = self._vertical_face_id(ix + 1, iy)
        f2 = self._horizontal_face_id(ix, iy + 1)
        f3 = self._vertical_face_id(ix, iy)
        self.cell_faceid[cell_index] = np.array([f0, f1, f2, f3, 4], dtype=np.int32)

        self._append_value(self.node_cellid, n0, cell_index)
        self._append_value(self.node_cellid, n1, cell_index)
        self._append_value(self.node_cellid, n2, cell_index)
        self._append_value(self.node_cellid, n3, cell_index)

        self._set_cell_face_neighbors(cell_index, ix, iy)
        self._set_cell_node_neighbors(cell_index, ix, iy)
        cell_index += 1

  def _set_cell_face_neighbors(self, cell_index, ix, iy):
    face_neighbors = []
    if iy - 1 >= 0:
      face_neighbors.append(self._cell_id(ix, iy - 1))
    if iy + 1 < self.width:
      face_neighbors.append(self._cell_id(ix, iy + 1))
    if ix + 1 < self.width:
      face_neighbors.append(self._cell_id(ix + 1, iy))
    if ix - 1 >= 0:
      face_neighbors.append(self._cell_id(ix - 1, iy))

    self.cell_cellfid[cell_index, 0:len(face_neighbors)] = face_neighbors
    self.cell_cellfid[cell_index, -1] = len(face_neighbors)

  def _set_cell_node_neighbors(self, cell_index, ix, iy):
    node_neighbors = []
    if iy - 1 >= 0:
      node_neighbors.append(self._cell_id(ix, iy - 1))
    if iy + 1 < self.width:
      node_neighbors.append(self._cell_id(ix, iy + 1))
    if ix + 1 < self.width:
      node_neighbors.append(self._cell_id(ix + 1, iy))
    if ix - 1 >= 0:
      node_neighbors.append(self._cell_id(ix - 1, iy))
    if ix + 1 < self.width and iy - 1 >= 0:
      node_neighbors.append(self._cell_id(ix + 1, iy - 1))
    if ix - 1 >= 0 and iy - 1 >= 0:
      node_neighbors.append(self._cell_id(ix - 1, iy - 1))
    if ix + 1 < self.width and iy + 1 < self.width:
      node_neighbors.append(self._cell_id(ix + 1, iy + 1))
    if ix - 1 >= 0 and iy + 1 < self.width:
      node_neighbors.append(self._cell_id(ix - 1, iy + 1))

    self.cell_cellnid[cell_index, 0:len(node_neighbors)] = node_neighbors
    self.cell_cellnid[cell_index, -1] = len(node_neighbors)

  def _build_horizontal_faces(self):
    # Horizontal faces connect nodes with the same y-index in the logical grid.
    face_index = 0
    for ix in range(self.width):
      for iy in range(self.width + 1):
        n0 = self._node_id(ix, iy)
        n1 = self._node_id(ix + 1, iy)
        self.faces[face_index] = np.array([n0, n1, 2], dtype=np.int32)

        if iy == 0:
          self.face_cellid[face_index] = np.array([self._cell_id(ix, 0), -1], dtype=np.int32)
          self.face_oldname[face_index] = 3
        elif iy == self.width:
          self.face_cellid[face_index] = np.array([self._cell_id(ix, self.width - 1), -1], dtype=np.int32)
          self.face_oldname[face_index] = 4
        else:
          self.face_cellid[face_index] = np.array([self._cell_id(ix, iy), self._cell_id(ix, iy - 1)], dtype=np.int32)
          self.face_oldname[face_index] = 0

        face_index += 1
    return face_index

  def _build_vertical_faces(self, face_index):
    # Vertical faces continue after all horizontal faces in the global numbering.
    for ix in range(self.width + 1):
      for iy in range(self.width):
        n0 = self._node_id(ix, iy)
        n1 = self._node_id(ix, iy + 1)
        self.faces[face_index] = np.array([n0, n1, 2], dtype=np.int32)

        if ix == 0:
          self.face_cellid[face_index] = np.array([self._cell_id(0, iy), -1], dtype=np.int32)
          self.face_oldname[face_index] = 1
        elif ix == self.width:
          self.face_cellid[face_index] = np.array([self._cell_id(self.width - 1, iy), -1], dtype=np.int32)
          self.face_oldname[face_index] = 2
        else:
          self.face_cellid[face_index] = np.array([self._cell_id(ix, iy), self._cell_id(ix - 1, iy)], dtype=np.int32)
          self.face_oldname[face_index] = 0

        face_index += 1

  def _register_phy_face(self, phy_index, face_id):
    # A physical face is a boundary face. We keep both directions of the map.
    self.phy_faces[phy_index] = self.faces[face_id]
    self.face_to_phyid[face_id] = phy_index

    oldname = self.face_oldname[face_id]
    for nodeid in self.faces[face_id, 0:2]:
      self._append_value(self.node_ghostid, nodeid, phy_index)
      if self.node_oldname[nodeid] == 0 or oldname < self.node_oldname[nodeid]:
        self.node_oldname[nodeid] = oldname

  def _build_boundary_faces(self):
    # Boundary order is down, right, up, left. This matches the old manual layout.
    phy_index = 0

    for ix in range(self.width):
      face_id = self._horizontal_face_id(ix, 0)
      self._register_phy_face(phy_index, face_id)
      phy_index += 1

    for iy in range(self.width):
      face_id = self._vertical_face_id(self.width, iy)
      self._register_phy_face(phy_index, face_id)
      phy_index += 1

    for ix in range(self.width):
      face_id = self._horizontal_face_id(ix, self.width)
      self._register_phy_face(phy_index, face_id)
      phy_index += 1

    for iy in range(self.width):
      face_id = self._vertical_face_id(0, iy)
      self._register_phy_face(phy_index, face_id)
      phy_index += 1

  def _build_cell_ghostnid(self):
    # A cell ghost list is the set of physical faces touching any of its nodes.
    for cell_index in range(self.nb_cells):
      for nodeid in self.cells[cell_index, 0:4]:
        node_ghost_count = self.node_ghostid[nodeid, -1]
        for j in range(node_ghost_count):
          phyid = self.node_ghostid[nodeid, j]
          self._append_unique_value(self.cell_ghostnid, cell_index, phyid)

  def _set_phy_id_to_face_id(self):
    mask = self.face_to_phyid != -1
    self.phy_id_to_face_id[self.face_to_phyid[mask]] = np.nonzero(mask)[0]

  def create_mesh(self):
    """
    Build the topological tables of the structured rectangle mesh.
    """
    self._build_nodes()
    self._build_cells()
    face_index = self._build_horizontal_faces()
    self._build_vertical_faces(face_index)
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
    ghost_info_int # (0=cell_id, 1=face index inside the cell, 2=face_oldname, 3=cell global id)
    """
    for i in range(self.nb_cells):
      node_ids = self.cells[i, 0:self.cells[i, -1]]
      vertices = self.nodes[node_ids]
      self.cell_center[i] = np.sum(vertices, axis=0) / 4.0
      self.cell_volume[i] = self.x_length * self.y_length

    for i in range(self.nb_faces):
      node_ids = self.faces[i, 0:self.faces[i, -1]]
      vertices = self.nodes[node_ids]
      center = np.sum(vertices, axis=0) / 2.0
      tangent = vertices[1] - vertices[0]
      normal = np.array([-tangent[1], tangent[0], 0.0], dtype=np.float64)

      self.face_center[i] = center
      self.face_normal[i] = normal
      self.face_measure[i] = np.sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1])

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




