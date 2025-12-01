from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.backends.types import FLOAT_TYPE
import numpy as np
from manapy.tests.meshes import get_mesh
from manapy.tests.helpers.DomainTables import DomainTables
import numpy as np
from manapy.tests.helpers.TestLogger import TestLogger
from manapy.tests.helpers.GeneralTestTables import GeneralTestTables
import pickle
import os
import numba


def create_domain(nb_parts, mesh_path, dim):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  if nb_parts > 1:
    partitioning.make_n_part_old(nb_parts)
  local_domain_data = partitioning.create_sub_domains()

  if nb_parts == 1:
    local_domain_data[0].cell_loctoglob = np.arange(len(partitioning.cells), dtype=np.int32)
    local_domain_data[0].node_loctoglob = np.arange(len(partitioning.nodes), dtype=np.int32)

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return SingleCoreDomainTables(domains, FLOAT_TYPE)


# @numba.njit()
def _reinterpret_float32_as_int32(i):
  return np.float32(i)


# @numba.njit()
def node_helper(n_node_ghostcenter, n_node_haloghostcenter, o_node_haloghostcenter, n_halo_halosext, o_halo_halosext,
                dim, nb_parts):
  for i in range(n_node_ghostcenter.shape[0]):
    for j in range(n_node_ghostcenter.shape[1]):
      item = n_node_ghostcenter[i, j]
      if dim == 2 and item[0] != -1:
        item[2] = _reinterpret_float32_as_int32(item[2])
        item[3] = _reinterpret_float32_as_int32(item[3])
        item[4] = _reinterpret_float32_as_int32(item[4])
      if dim == 3 and item[0] != -1:
        item[3] = _reinterpret_float32_as_int32(item[3])
        item[4] = _reinterpret_float32_as_int32(item[4])
        item[5] = _reinterpret_float32_as_int32(item[5])

  if nb_parts != 1:
    for i in range(n_node_haloghostcenter.shape[0]):
      for j in range(n_node_haloghostcenter.shape[1]):
        item = n_node_haloghostcenter[i, j]
        if dim == 2 and item[0] != -1:
          item[2] = _reinterpret_float32_as_int32(item[2])
          item[3] = _reinterpret_float32_as_int32(item[3])
          item[4] = _reinterpret_float32_as_int32(item[4])
          item[2] = n_halo_halosext[int(item[2])][0]
        if dim == 3 and item[0] != -1:
          item[3] = _reinterpret_float32_as_int32(item[3])
          item[4] = _reinterpret_float32_as_int32(item[4])
          item[5] = _reinterpret_float32_as_int32(item[5])
          item[3] = n_halo_halosext[int(item[3])][0]

    for i in range(o_node_haloghostcenter.shape[0]):
      for j in range(o_node_haloghostcenter.shape[1]):
        item = o_node_haloghostcenter[i, j]
        if dim == 2 and item[0] != -1:
          item[2] = o_halo_halosext[int(item[2])][0]
        if dim == 3 and item[0] != -1:
          item[3] = o_halo_halosext[int(item[3])][0]


class CompareDomains:

  def __init__(self, decimal_precision, nb_parts, dim, mesh_path, mesh_name):

    # Load self.nd and self.od
    cache_file = os.path.join(os.path.dirname(__file__), "results", f"{mesh_name}_{nb_parts}_{dim}.pkl")
    if os.path.exists(cache_file):
      with open(cache_file, "rb") as f:
        self.nd, self.od, self.g_nd, self.g_od = pickle.load(f)
    else:
      self.nd = create_domain(nb_parts, mesh_path, dim)
      self.od = DomainTables(nb_partitions=nb_parts, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)
      self.g_nd = create_domain(1, mesh_path, dim)
      self.g_od = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)
      with open(cache_file, "wb") as f:
        pickle.dump((self.nd, self.od, self.g_nd, self.g_od), f)

    # self.nd = create_domain(nb_parts, mesh_path, dim)
    # self.g_nd = create_domain(1, mesh_path, dim)

    self.nb_parts = nb_parts
    self.dim = dim
    self.float_precision = FLOAT_TYPE
    self.decimal_precision = decimal_precision
    self.logger = TestLogger()

  def summary(self):
    return self.logger.summary()

  @staticmethod
  def sort_float_arr(arr, size):
    # Lexicographic sort by rows.
    if size == 3:
      arr = np.round(arr, decimals=3)  # np.round to limit sort precision sometime 10.5 is bigger than 10.5
      keys = [arr[:, 0], arr[:, 1], arr[:, 2]]
      indices = np.lexsort(keys)
      arr = arr[indices]
    elif size == 2:
      arr = np.round(arr, decimals=2)  # np.round to limit sort precision sometime 10.5 is bigger than 10.5
      keys = [arr[:, 0], arr[:, 1]]
      indices = np.lexsort(keys)
      arr = arr[indices]
    else:
      raise ValueError("size must be 2 or 3")
    return arr

  def test_cell_info(self):
    for p in range(self.nb_parts):
      o_nb_cells = self.od.d_cells[p].shape[0]
      n_nb_cells = self.nd.d_cells[p].shape[0]

      o_nodes = self.od.d_nodes[p][:, 0:3]
      n_nodes = self.nd.d_nodes[p]

      self.od.d_face_halofid[p][self.od.d_face_name[p] != 10] = -1
      # Number of cells
      self.logger.testing("Number of cells", np.testing.assert_equal, o_nb_cells, n_nb_cells)

      for i in range(o_nb_cells):
        # Vertices
        n_cell = self.nd.d_cells[p][i]
        n_cell_nodes = n_cell[0:n_cell[-1]]
        n_vertices = n_nodes[n_cell_nodes]

        o_cell = self.od.d_cells[p][i]
        o_cell_nodes = o_cell[0:o_cell[-1]]
        o_vertices = o_nodes[o_cell_nodes]
        self.logger.testing("Cell Vertices", np.testing.assert_almost_equal, n_vertices, o_vertices,
                            self.decimal_precision)

        # Cell Center
        n_center = self.nd.d_cell_center[p][i]
        o_center = self.od.d_cell_center[p][i]
        self.logger.testing("Cell Vertices", np.testing.assert_almost_equal, n_center, o_center,
                            self.decimal_precision)

        # Cell Area
        n_area = self.nd.d_cell_volume[p][i]
        o_area = self.od.d_cell_volume[p][i]
        self.logger.testing("Cell Area", np.testing.assert_almost_equal, n_area, o_area,
                            self.decimal_precision)

        # Neighbors by face
        n_cellfid = self.nd.d_cell_cellfid[p][i]
        n_cellfid = np.sort(n_cellfid[0:n_cellfid[-1]])
        o_cellfid = self.od.d_cell_cellfid[p][i]
        o_cellfid = np.sort(o_cellfid[0:o_cellfid[-1]])
        self.logger.testing("Cell Neighbors by face", np.testing.assert_equal, n_cellfid, o_cellfid)

        # Neighbors by node
        n_cellnid = self.nd.d_cell_cellnid[p][i]
        n_cellnid = np.sort(n_cellnid[0:n_cellnid[-1]])
        o_cellnid = self.od.d_cell_cellnid[p][i]
        o_cellnid = np.sort(o_cellnid[0:o_cellnid[-1]])
        self.logger.testing("Cell Neighbors by node", np.testing.assert_equal, n_cellnid, o_cellnid)

        # Halo by face
        if self.nb_parts != 1:
          n_cell_faces = self.nd.d_cell_faces[p][i]
          n_cell_faces = n_cell_faces[0:n_cell_faces[-1]]
          n_halofid = self.nd.d_face_halofid[p][n_cell_faces]  # get cell halo cells
          n_halofid = n_halofid[n_halofid != -1]  # get cell halo cells
          n_halofid = self.nd.d_halo_halosext[p][n_halofid][:, 0]  # get halos global index
          n_halofid = np.sort(n_halofid)

          o_cell_faces = self.od.d_cell_faces[p][i]
          o_cell_faces = o_cell_faces[0:o_cell_faces[-1]]
          o_halofid = self.od.d_face_halofid[p][o_cell_faces]  # get cell halo cells
          o_halofid = o_halofid[o_halofid != -1]  # get cell halo cells
          o_halofid = self.od.d_halo_halosext[p][o_halofid][:, 0]  # get halos global index
          o_halofid = np.sort(o_halofid)

          self.logger.testing("Cell Halo by face", np.testing.assert_equal, n_halofid, o_halofid)

        # Halo by node
        if self.nb_parts != 1:
          n_halonid = self.nd.d_cell_halonid[p][i]
          n_halonid = self.nd.d_halo_halosext[p][n_halonid[0:n_halonid[-1]]][:, 0]  # get domain global halo cells index
          n_halonid = np.sort(n_halonid)

          o_halonid = self.od.d_cell_halonid[p][i]
          o_halonid = self.od.d_halo_halosext[p][o_halonid[0:o_halonid[-1]]][:, 0]  # get domain global halo cells index
          o_halonid = np.sort(o_halonid)
          self.logger.testing("Cell Halo by node", np.testing.assert_equal, n_halonid, o_halonid)

        # Ghostnid And face_ghostcenter and gamma
        n_ghostnid = self.nd.d_cell_ghostnid[p][i]
        n_ghostnid = n_ghostnid[0:n_ghostnid[-1]]
        n_ghostn_center = self.nd.d_face_ghostcenter[p][n_ghostnid]
        n_ghostn_center = self.sort_float_arr(n_ghostn_center, self.dim)

        o_ghostnid = self.od.d_cell_ghostnid[p][i]
        o_ghostnid = o_ghostnid[0:o_ghostnid[-1]]
        o_ghostn_center = self.od.d_face_ghostcenter[p][o_ghostnid]
        o_ghostn_center = self.sort_float_arr(o_ghostn_center, self.dim)

        self.logger.testing("Ghostnid And face_ghostcenter", np.testing.assert_almost_equal,
                            n_ghostn_center[:, 0:self.dim], o_ghostn_center[:, 0:self.dim], self.decimal_precision)
        self.logger.testing("ghost Gamma", np.testing.assert_almost_equal, n_ghostn_center[:, self.dim],
                            o_ghostn_center[:, self.dim], self.decimal_precision)

        # Haloghostnid and Haloghostcenter
        if self.nb_parts != 1:
          n_haloghostnid = self.nd.d_cell_haloghostnid[p][i]
          n_haloghostnid = n_haloghostnid[0:n_haloghostnid[-1]]
          n_haloghostcenter = self.nd.d_cell_haloghostcenter[p][n_haloghostnid]
          n_haloghostcenter = self.sort_float_arr(n_haloghostcenter, self.dim)

          o_haloghostnid = self.od.d_cell_haloghostnid[p][i]
          o_haloghostnid = o_haloghostnid[0:o_haloghostnid[-1]]
          o_haloghostcenter = self.od.d_cell_haloghostcenter[p][o_haloghostnid]
          o_haloghostcenter = self.sort_float_arr(o_haloghostcenter, self.dim)
          self.logger.testing("Cell Haloghostnid and Haloghostcenter *", np.testing.assert_almost_equal,
                              n_haloghostcenter, o_haloghostcenter, self.decimal_precision)

  def test_node_info(self):

    # Number of nodes
    for i in range(self.g_nd.d_nodes[0].shape[0]):
      n_node = self.g_nd.d_nodes[0][i]
      o_node = self.g_od.d_nodes[0][i][0:-1]
      self.logger.testing("Node Vertex 1 (g_new to g_old)", np.testing.assert_almost_equal, n_node, o_node, 5)

    for p in range(self.nb_parts):
      # Pre processing
      n_node_ghostcenter = self.nd.d_node_ghostcenter[p]
      n_node_haloghostcenter = self.nd.d_node_haloghostcenter[p]
      o_node_haloghostcenter = self.od.d_node_haloghostcenter[p]
      node_helper(n_node_ghostcenter, n_node_haloghostcenter, o_node_haloghostcenter, self.nd.d_halo_halosext[p],
                  self.od.d_halo_halosext[p], self.dim, self.nb_parts)

      n_nb_cells = self.nd.d_cells[p].shape[0]
      o_nb_cells = self.od.d_cells[p].shape[0]
      n_nb_nodes = self.nd.d_nodes[p].shape[0]
      o_nb_nodes = self.od.d_nodes[p].shape[0]
      self.logger.testing("Number of nodes", np.testing.assert_equal, o_nb_nodes, n_nb_nodes)

      for i in range(n_nb_nodes):
        g_index = self.nd.d_node_loctoglob[p][i]
        n_node = self.nd.d_nodes[p][i]
        o_node = self.g_od.d_nodes[0][g_index][0:-1]
        self.logger.testing("Node Vertex 2 (new to g_old)", np.testing.assert_almost_equal, n_node, o_node, 5)

      for i in range(o_nb_nodes):
        g_index = self.od.d_node_loctoglob[p][i]
        o_node = self.od.d_nodes[p][i][0:-1]
        n_node = self.g_nd.d_nodes[0][g_index]
        self.logger.testing("Node Vertex 3 (old to g_new)", np.testing.assert_almost_equal, n_node, o_node, 5)

      for i in range(n_nb_cells):
        n_cell_nodes = self.nd.d_cells[p][i]
        o_cell_nodes = self.od.d_cells[p][i]

        self.logger.testing("Cell nb Nodes", np.testing.assert_equal, n_cell_nodes[-1], o_cell_nodes[-1])
        for j in range(n_cell_nodes[-1]):
          # Cellid
          n_nodeid = self.nd.d_cells[p][i, j]
          n_node_cellid = self.nd.d_node_cellid[p][n_nodeid][0:self.nd.d_node_cellid[p][n_nodeid, -1]]
          n_node_cellid = self.nd.d_cell_loctoglob[p][n_node_cellid]
          n_node_cellid = np.sort(n_node_cellid)

          o_nodeid = self.od.d_cells[p][i, j]
          o_node_cellid = self.od.d_node_cellid[p][o_nodeid][0:self.od.d_node_cellid[p][o_nodeid, -1]]
          o_node_cellid = self.od.d_cell_loctoglob[p][o_node_cellid]
          o_node_cellid = np.sort(o_node_cellid)
          self.logger.testing("Node Cellid", np.testing.assert_equal, n_node_cellid, o_node_cellid)

          # Halonid
          if self.nb_parts != 1:
            n_node_halonid = self.nd.d_node_halonid[p][n_nodeid]
            n_node_halonid = n_node_halonid[0:n_node_halonid[-1]]
            n_node_halonid = self.nd.d_halo_halosext[p][n_node_halonid][:, 0]
            n_node_halonid = np.sort(n_node_halonid)

            o_node_halonid = self.od.d_node_halonid[p][o_nodeid]
            o_node_halonid = o_node_halonid[0:o_node_halonid[-1]]
            o_node_halonid = self.od.d_halo_halosext[p][o_node_halonid][:, 0]
            o_node_halonid = np.sort(o_node_halonid)
            self.logger.testing("Node Halonid", np.testing.assert_equal, n_node_halonid, o_node_halonid)

          # Node Loctoglob
          n_node_g_id = self.nd.d_node_loctoglob[p][n_nodeid]
          o_node_g_id = self.od.d_node_loctoglob[p][o_nodeid]
          self.logger.testing("Node Loctoglob", np.testing.assert_equal, n_node_g_id, o_node_g_id)

          # Node Oldname
          n_node_oldname = self.nd.d_node_oldname[p][n_nodeid]
          o_node_oldname = self.od.d_node_oldname[p][o_nodeid]
          self.logger.testing("Node Oldname", np.testing.assert_equal, n_node_oldname, o_node_oldname)

          # Node Name
          n_node_name = self.nd.d_node_name[p][n_nodeid]
          o_node_name = self.od.d_node_name[p][o_nodeid]
          self.logger.testing("Node Name", np.testing.assert_equal, n_node_name, o_node_name)

          # Node: ghostid, ghostcenter, ghostfaceinfo, d_face_ghostcenter
          """
          2D
          * node_ghostid # [indices point to faces aka faceid]
          * node_ghostcenter  # [ghost_center x.y, cell_id, face_old_name, face_id]
          * face_ghostcenter  # [ghost_center x.y, gamma]
          * node_ghostfaceinfo  # [face_center x.y, face_normal x.y]

          3D
          * node_ghostid # [indices point to faces aka faceid]
          * node_ghostcenter  # [ghost_center x.y.z, cell_id, face_old_name, face_id]
          * face_ghostcenter  # [ghost_center x.y.z, gamma]
          * node_ghostfaceinfo  # [face_center x.y.z, face_normal x.y.z]
          """
          n_node_ghostcenter = self.nd.d_node_ghostcenter[p][n_nodeid]
          n_node_ghostcenter_face_id = np.int32(n_node_ghostcenter[n_node_ghostcenter[:, 0] != -1][:, -1])
          n_node_ghostcenter_face_id = self.nd.d_face_center[p][n_node_ghostcenter_face_id]
          n_node_ghostfaceinfo = self.nd.d_node_ghostfaceinfo[p][n_nodeid]
          n_node_ghostid = self.nd.d_node_ghostid[p][n_nodeid]
          n_node_ghostid = n_node_ghostid[0:n_node_ghostid[-1]]
          n_face_ghostcenter = self.nd.d_face_ghostcenter[p][n_node_ghostid]

          o_node_ghostcenter = self.od.d_node_ghostcenter[p][o_nodeid]
          o_node_ghostcenter_face_id = []
          for k in range(o_node_ghostcenter.shape[0]):
            if o_node_ghostcenter[k, 0] != -1:
              o_node_ghostcenter_face_id.append(int(o_node_ghostcenter[k, -1]))
          o_node_ghostcenter_face_id = self.od.d_face_center[p][o_node_ghostcenter_face_id]
          o_node_ghostfaceinfo = self.od.d_node_ghostfaceinfo[p][o_nodeid]
          o_node_ghostid = self.od.d_node_ghostid[p][o_nodeid]
          o_node_ghostid = o_node_ghostid[0:o_node_ghostid[-1]]
          o_face_ghostcenter = self.od.d_face_ghostcenter[p][o_node_ghostid]

          n_node_ghostcenter = self.sort_float_arr(n_node_ghostcenter, self.dim)
          n_node_ghostcenter_face_id = self.sort_float_arr(n_node_ghostcenter_face_id, self.dim)
          n_node_ghostfaceinfo = self.sort_float_arr(n_node_ghostfaceinfo, self.dim)
          n_face_ghostcenter = self.sort_float_arr(n_face_ghostcenter, self.dim)
          o_node_ghostcenter = self.sort_float_arr(o_node_ghostcenter, self.dim)
          o_node_ghostcenter_face_id = self.sort_float_arr(o_node_ghostcenter_face_id, self.dim)
          o_node_ghostfaceinfo = self.sort_float_arr(o_node_ghostfaceinfo, self.dim)
          o_face_ghostcenter = self.sort_float_arr(o_face_ghostcenter, self.dim)
          self.logger.testing("Node node_ghostcenter", np.testing.assert_almost_equal,
                              n_node_ghostcenter[:, 0:-1], o_node_ghostcenter[:, 0:-1], decimal=self.decimal_precision)
          self.logger.testing("Node node_ghostcenter (face_id)", np.testing.assert_almost_equal,
                              n_node_ghostcenter_face_id, o_node_ghostcenter_face_id, decimal=self.decimal_precision)
          self.logger.testing("Node ghostfaceinfo", np.testing.assert_almost_equal,
                              n_node_ghostfaceinfo, o_node_ghostfaceinfo, decimal=self.decimal_precision)
          self.logger.testing("Node face_ghostcenter and node_ghostid", np.testing.assert_almost_equal,
                              n_face_ghostcenter[:, 0:self.dim], o_face_ghostcenter[:, 0:self.dim],
                              decimal=self.decimal_precision)
          self.logger.testing("Node face_ghostcenter gamma", np.testing.assert_almost_equal,
                              n_face_ghostcenter[:, self.dim], o_face_ghostcenter[:, self.dim],
                              decimal=self.decimal_precision)
          # Node: haloghostid, haloghostcenter, haloghostfaceinfo, d_cell_haloghostcenter
          """
          2D
          * cell_haloghostcenter [[g_x, g_y, unused g_z]]
          * node_haloghostid [[indices point to cell_haloghostcenter]]
          * node_haloghostcenter [[[g_x, g_y, (halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
          * node_haloghostfaceinfo [[[fc_x, fc_y, fn_x, fn_y]]]

          3D
          * cell_haloghostcenter [[g_x, g_y, g_z]]
          * node_haloghostid [[indices point to cell_haloghostcenter]]
          * node_haloghostcenter [[[g_x, g_y, g_z, (halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
          * node_haloghostfaceinfo [[[fc_x, fc_y, fc_z, fn_x, fn_y, fn_z]]]
          """
          if self.nb_parts != 1:
            n_node_haloghostcenter = self.nd.d_node_haloghostcenter[p][n_nodeid]
            n_node_haloghostcenter = n_node_haloghostcenter[:, 0:-1]
            n_node_haloghostfaceinfo = self.nd.d_node_haloghostfaceinfo[p][n_nodeid]
            n_node_haloghostid = self.nd.d_node_haloghostid[p][n_nodeid]
            n_node_haloghostid = n_node_haloghostid[0:n_node_haloghostid[-1]]
            n_cell_haloghostcenter = self.nd.d_cell_haloghostcenter[p][n_node_haloghostid]

            o_node_haloghostcenter = self.od.d_node_haloghostcenter[p][o_nodeid]
            o_node_haloghostcenter = o_node_haloghostcenter[:, 0:-1]
            o_node_haloghostfaceinfo = self.od.d_node_haloghostfaceinfo[p][o_nodeid]
            o_node_haloghostid = self.od.d_node_haloghostid[p][o_nodeid]
            o_node_haloghostid = o_node_haloghostid[0:o_node_haloghostid[-1]]
            o_cell_haloghostcenter = self.od.d_cell_haloghostcenter[p][o_node_haloghostid]

            n_node_haloghostcenter = self.sort_float_arr(n_node_haloghostcenter, self.dim)
            n_node_haloghostfaceinfo = self.sort_float_arr(n_node_haloghostfaceinfo, self.dim)
            n_cell_haloghostcenter = self.sort_float_arr(n_cell_haloghostcenter, self.dim)
            o_node_haloghostcenter = self.sort_float_arr(o_node_haloghostcenter, self.dim)
            o_node_haloghostfaceinfo = self.sort_float_arr(o_node_haloghostfaceinfo, self.dim)
            o_cell_haloghostcenter = self.sort_float_arr(o_cell_haloghostcenter, self.dim)
            self.logger.testing("Node node_haloghostcenter", np.testing.assert_almost_equal,
                                n_node_haloghostcenter, o_node_haloghostcenter, decimal=self.decimal_precision)
            self.logger.testing("Node node_haloghostfaceinfo", np.testing.assert_almost_equal,
                                n_node_haloghostfaceinfo, o_node_haloghostfaceinfo, decimal=self.decimal_precision)
            self.logger.testing("Node cell_haloghostcenter and node_haloghostid", np.testing.assert_almost_equal,
                                n_cell_haloghostcenter, o_cell_haloghostcenter, decimal=self.decimal_precision)

  def test_face_info(self):
    # order of the face inside the cell is different
    # order of the nodes inside the face is different

    for p in range(self.nb_parts):
      for i in range(self.od.d_cells[p].shape[0]):
        n_c_faces = self.nd.d_cell_faces[p][i]
        n_c_faces = n_c_faces[0:n_c_faces[-1]]
        o_c_faces = self.od.d_cell_faces[p][i]
        o_c_faces = o_c_faces[0:o_c_faces[-1]]

        n_faces = self.nd.d_faces[p]
        o_faces = self.od.d_faces[p]
        n_node_loctoglob = self.nd.d_node_loctoglob[p]
        o_node_loctoglob = self.od.d_node_loctoglob[p]
        a = [list(np.sort(n_node_loctoglob[n_faces[x, 0:n_faces[x, -1]]])) for x in n_c_faces]
        b = [list(np.sort(o_node_loctoglob[o_faces[x, 0:o_faces[x, -1]]])) for x in o_c_faces]
        indices = [b.index(x) for x in a]
        o_c_faces = o_c_faces[indices]

        # Vertices
        for j in range(n_c_faces.shape[0]):
          n_face = self.nd.d_faces[p][n_c_faces[j]]
          n_face_vertices = self.nd.d_nodes[p][n_face[0:n_face[-1]]]
          n_face_vertices = self.sort_float_arr(n_face_vertices, self.dim)
          o_face = self.od.d_faces[p][o_c_faces[j]]
          o_face_vertices = self.od.d_nodes[p][o_face[0:o_face[-1]]][:, 0:3]
          o_face_vertices = self.sort_float_arr(o_face_vertices, self.dim)
          self.logger.testing("Face Vertices", np.testing.assert_almost_equal, n_face_vertices, o_face_vertices,
                              self.decimal_precision)

        # Measure
        n_faces_measure = self.nd.d_face_measure[p][n_c_faces]
        o_faces_measure = self.od.d_face_measure[p][o_c_faces]
        self.logger.testing("Face Measure", np.testing.assert_almost_equal, n_faces_measure, o_faces_measure,
                            self.decimal_precision)

        # Face center
        n_faces_center = self.nd.d_face_center[p][n_c_faces]
        o_faces_center = self.od.d_face_center[p][o_c_faces]
        self.logger.testing("Face Center", np.testing.assert_almost_equal, n_faces_center, o_faces_center,
                            self.decimal_precision)

        # Name
        n_faces_name = self.nd.d_face_name[p][n_c_faces]
        o_faces_name = self.od.d_face_name[p][o_c_faces]
        self.logger.testing("Face Name", np.testing.assert_equal, n_faces_name, o_faces_name)

        # Oldname
        n_faces_oldname = self.nd.d_face_oldname[p][n_c_faces]
        o_faces_oldname = self.od.d_face_oldname[p][o_c_faces]
        o_faces_oldname[o_faces_oldname == 10] = 0
        self.logger.testing("Face Oldname", np.testing.assert_equal, n_faces_oldname, o_faces_oldname)

        # ! Normal (Only abs)
        n_faces_normal = np.abs(self.nd.d_face_normal[p][n_c_faces])
        o_faces_normal = np.abs(self.od.d_face_normal[p][o_c_faces])
        self.logger.testing("Face Normal(Only abs)", np.testing.assert_almost_equal, n_faces_normal, o_faces_normal,
                            self.decimal_precision)

        # CellId
        n_faces_cellid = self.nd.d_face_cellid[p][n_c_faces]
        tmp = self.nd.d_cell_loctoglob[p][n_faces_cellid]
        tmp[n_faces_cellid == -1] = -1
        n_faces_cellid = np.sort(tmp)

        o_faces_cellid = self.od.d_face_cellid[p][o_c_faces]
        o_faces_cellid[o_faces_cellid < 0] = -1
        tmp = self.od.d_cell_loctoglob[p][o_faces_cellid]
        tmp[o_faces_cellid == -1] = -1
        o_faces_cellid = np.sort(tmp)
        self.logger.testing("Face CellId", np.testing.assert_equal, n_faces_cellid, o_faces_cellid)

        # Ghostcenter
        c_faces_ghostcenter = self.nd.d_face_ghostcenter[p][n_c_faces]
        o_faces_ghostcenter = self.od.d_face_ghostcenter[p][o_c_faces]
        self.logger.testing("Face: Ghostcenter", np.testing.assert_almost_equal, c_faces_ghostcenter[:, 0:self.dim],
                            o_faces_ghostcenter[:, 0:self.dim], self.decimal_precision)
        self.logger.testing("Face: Ghostcenter gamma", np.testing.assert_almost_equal, c_faces_ghostcenter[:, self.dim],
                            o_faces_ghostcenter[:, self.dim], self.decimal_precision)
        if self.dim == 2:
          # Cell face normal (cells.cell_nf) # TODO cells.cell_nf not used on 3D
          n_cell_nf = self.nd.d_cell_nf[p][i]
          o_cell_nf = self.od.d_cell_nf[p][i]
          self.logger.testing("Face Cell face normal", np.testing.assert_almost_equal, n_cell_nf, o_cell_nf,
                              self.decimal_precision)

  def test_halo_info(self):
    if self.nb_parts <= 1:
      return

    for p in range(self.nb_parts):
      # halosext
      n_halosext = self.nd.d_halo_halosext[p]
      o_halosext = self.od.d_halo_halosext[p]
      n_indcies = np.argsort(n_halosext[:, 0])
      o_indcies = np.argsort(o_halosext[:, 0])
      n_halosext = n_halosext[n_indcies]
      o_halosext = o_halosext[o_indcies]
      self.logger.testing("Halo Number of Halosext", np.testing.assert_equal, n_halosext.shape[0], o_halosext.shape[0])
      for i in range(n_halosext.shape[0]):
        n_halosext_ele = n_halosext[i, 0:n_halosext[i, -1]]
        o_halosext_ele = o_halosext[i, 0:o_halosext[i, -1]]
        self.logger.testing("Halo Halosext", np.testing.assert_equal, n_halosext_ele, o_halosext_ele)

      # Halosint and halo_neigh
      start = 0
      n_dic = {}
      for neigh in range(self.nd.d_halo_neigh[p].shape[1]):
        neigh_part = self.nd.d_halo_neigh[p][0, neigh]
        nb_haloint = self.nd.d_halo_neigh[p][1, neigh]
        c_haloint = self.nd.d_halo_halosint[p][start:start + nb_haloint]
        c_haloint = self.nd.d_cell_loctoglob[p][c_haloint]
        # c_haloint = np.sort(self.nd.d_cell_loctoglob[p][c_haloint])
        start += nb_haloint
        n_dic[neigh_part] = c_haloint

      start = 0
      o_dic = {}
      for neigh in range(self.od.d_halo_neigh[p].shape[1]):
        neigh_part = self.od.d_halo_neigh[p][0, neigh]
        nb_haloint = self.od.d_halo_neigh[p][1, neigh]
        c_haloint = self.od.d_halo_halosint[p][start:start + nb_haloint]
        start += nb_haloint
        o_dic[neigh_part] = c_haloint

      n_keys = np.sort(np.array(list(n_dic.keys()), dtype=np.int32))
      o_keys = np.sort(np.array(list(o_dic.keys()), dtype=np.int32))
      self.logger.testing(f"Halo Halosint and halo_neigh keys", np.testing.assert_equal, n_keys, o_keys)
      for key in n_dic.keys():
        print(n_dic[key])
        self.logger.testing(f"Halo Halosint and halo_neigh", np.testing.assert_equal, n_dic[key], o_dic[key])

      # Halo: centvol
      n_halo_centvol = self.nd.d_halo_centvol[p]
      o_halo_centvol = self.od.d_halo_centvol[p]
      n_halo_centvol = n_halo_centvol[n_indcies]
      o_halo_centvol = o_halo_centvol[o_indcies]
      for i in range(o_halo_centvol.shape[0]):
        self.logger.testing("Halo center and vol", np.testing.assert_almost_equal, n_halo_centvol[i], o_halo_centvol[i],
                            decimal=self.decimal_precision)

      # Halo : sizehaloghost
      n_sizehaloghost = self.nd.d_halo_sizehaloghost[p]
      o_sizehaloghost = self.od.d_halo_sizehaloghost[p]
      self.logger.testing("Halo sizehaloghost", np.testing.assert_equal, n_sizehaloghost, o_sizehaloghost)


dim, mesh_path, mesh_name = get_mesh(4)
compare_domains = CompareDomains(decimal_precision=2, nb_parts=32, mesh_name=mesh_name, dim=dim, mesh_path=mesh_path)
compare_domains.test_cell_info()
compare_domains.test_node_info()
compare_domains.test_face_info()
compare_domains.test_halo_info()
compare_domains.summary()