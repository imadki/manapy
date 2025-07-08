import numpy as np
from .TestLogger import TestLogger

class TetraChecker3D:

  def __init__(self, decimal_precision, domain_tables, unified_domain, test_tables):

    self.nb_partitions = domain_tables.nb_partitions
    self.float_precision = domain_tables.float_precision
    self.domain_tables = domain_tables
    self.unified_domain = unified_domain
    self.decimal_precision = decimal_precision

    self.logger = TestLogger()
    self.test_tables = test_tables

  def summary(self):
    return self.logger.summary()

  def _sort_cell_faces(self, c_nodes, c_faces_nodes):
    faces_node = np.array([
      c_nodes[[0, 1, 2]],
      c_nodes[[0, 1, 3]],
      c_nodes[[0, 2, 3]],
      c_nodes[[1, 2, 3]]
    ], dtype=np.int32)

    tmp_faces_node = np.sort(faces_node, axis=1)
    tmp_c_faces_nodes = np.sort(c_faces_nodes, axis=1)

    matches = np.all(tmp_faces_node == tmp_c_faces_nodes[:, np.newaxis, :], axis=2)
    sorted_indexes = np.argmax(matches, axis=0)

    return faces_node, sorted_indexes

  def sort_float_arr(self, arr):
    # Lexicographic sort by rows.
    arr = np.round(arr, decimals=3) # np.round to limit sort precision sometime 10.5 is bigger than 10.5
    keys = [arr[:, 0], arr[:, 1], arr[:, 2]]
    indices = np.lexsort(keys)
    arr = arr[indices]
    return arr

  def test_cell_info(self):

    nb_cells = 0

    for p in range(self.nb_partitions):
      d_cell_loctoglob = self.domain_tables.d_cell_loctoglob[p]

      d_cells = self.domain_tables.d_cells[p]
      d_nodes = self.domain_tables.d_nodes[p][:, 0:3]
      d_cell_volume =self.domain_tables.d_cell_volume[p]
      d_cell_center = self.domain_tables.d_cell_center[p][:, 0:3]
      d_cell_cellfid = self.domain_tables.d_cell_cellfid[p]
      d_cell_cellnid = self.domain_tables.d_cell_cellnid[p]
      d_cell_faces = self.domain_tables.d_cell_faces[p]
      d_cell_halonid = self.domain_tables.d_cell_halonid[p]
      d_cell_ghostnid = self.domain_tables.d_cell_ghostnid[p]
      d_cell_haloghostcenter = self.domain_tables.d_cell_haloghostcenter[p]
      d_cell_haloghostnid = self.domain_tables.d_cell_haloghostnid[p]
      d_face_halofid = self.domain_tables.d_face_halofid[p]
      d_face_ghostcenter = self.domain_tables.d_face_ghostcenter[p]
      d_halo_halosext = self.domain_tables.d_halo_halosext[p]
      d_face_name = self.domain_tables.d_face_name[p]

      d_face_halofid[d_face_name != 10] = -1 # used by Halo by face section

      for i in range(len(d_cell_loctoglob)):
        g_index = d_cell_loctoglob[i]

        # Vertices
        cell_vertices = self.test_tables.cell_vertices[g_index]
        c_cell_vertices = d_nodes[d_cells[i][0:d_cells[i][-1]]]
        self.logger.testing("Cell Vertices", np.testing.assert_almost_equal, cell_vertices, c_cell_vertices, self.decimal_precision)

        # # Center
        # cell_center = self.test_tables.cell_center[g_index]
        # c_cell_center = d_cell_center[i]
        # self.logger.testing("Cell Center", np.testing.assert_almost_equal, cell_center, c_cell_center, self.decimal_precision)
        #
        # # Area
        # cell_area = self.test_tables.cell_area[g_index]
        # c_cell_area = d_cell_volume[i]
        # self.logger.testing("Cell Area", np.testing.assert_almost_equal, cell_area, c_cell_area, self.decimal_precision)
        #
        # # Neighbors by face
        # cellfid = self.test_tables.l_cell_cellfid[g_index]
        # cellfid = np.sort(cellfid[0:cellfid[-1]])
        # c_cellfid = d_cell_cellfid[i]
        # c_cellfid = np.sort(d_cell_loctoglob[c_cellfid[0:c_cellfid[-1]]])
        # self.logger.testing("Cell Neighbors by face", np.testing.assert_equal, cellfid, c_cellfid)
        #
        # # Neighbors by node
        # cellnid = self.test_tables.l_cell_cellnid[g_index]
        # cellnid = np.sort(cellnid[0:cellnid[-1]])
        # c_cellnid = d_cell_cellnid[i]
        # c_cellnid = np.sort(d_cell_loctoglob[c_cellnid[0:c_cellnid[-1]]])
        # self.logger.testing("Cell Neighbors by node", np.testing.assert_equal, cellnid, c_cellnid)

        # Halo by face
        # halofid = self.test_tables.cell_halofid[g_index]
        # halofid = np.sort(halofid[halofid != -1])
        # cell_faces = d_cell_faces[i] # get cell faces
        # c_halofid = d_face_halofid[cell_faces[0:cell_faces[-1]]] # get cell halo cells
        # c_halofid = c_halofid[c_halofid != -1] # get cell halo cells
        # c_halofid = d_halo_halosext[c_halofid][:, 0] # get halos global index
        # c_halofid = np.sort(c_halofid)
        # self.logger.testing("Cell Halo by face", np.testing.assert_equal, halofid, c_halofid)

        # Halo by node
        halonid = self.test_tables.cell_halonid[g_index]
        halonid = np.sort(halonid[halonid != -1])
        c_halonid = d_cell_halonid[i]
        c_halonid = d_halo_halosext[c_halonid[0:c_halonid[-1]]][:, 0] # get domain global halo cells index
        c_halonid = np.sort(c_halonid)
        self.logger.testing("Cell Halo by node", np.testing.assert_equal, halonid, c_halonid)

        # Ghostnid TODO Changed Need to check gamma
        ghostnid = self.test_tables.l_cell_ghostnid[g_index]
        ghostnid = ghostnid[0:ghostnid[-1]]
        ghostn_center = self.test_tables.ghost_info[ghostnid][:, 0:4]
        ghostn_center = self.sort_float_arr(ghostn_center)
        c_ghostnid = d_cell_ghostnid[i]
        c_ghostn_center = d_face_ghostcenter[c_ghostnid[0:c_ghostnid[-1]]]
        c_ghostn_center = self.sort_float_arr(c_ghostn_center)
        a = self.logger.testing("Cell Ghostnid *", np.testing.assert_almost_equal, c_ghostn_center[:, 0:3], ghostn_center[:, 0:3], self.decimal_precision)
        if not a:
          print("Cell Ghostnid *", p, g_index, i)
          return
        # Haloghostnid and Haloghostcenter
        haloghostnid = self.test_tables.cell_haloghostnid[g_index]
        haloghostnid = haloghostnid[0:haloghostnid[-1]]
        haloghostcenter = self.test_tables.ghost_info[haloghostnid][:, 0:3] #(center only)
        haloghostcenter = self.sort_float_arr(haloghostcenter)
        c_haloghostnid = d_cell_haloghostnid[i]
        c_haloghostnid = c_haloghostnid[0:c_haloghostnid[-1]]
        c_haloghostcenter = d_cell_haloghostcenter[c_haloghostnid][:, 0:3] # center (x, y, z)
        c_haloghostcenter = self.sort_float_arr(c_haloghostcenter)
        self.logger.testing("Cell Haloghostnid and Haloghostcenter *", np.testing.assert_almost_equal, c_haloghostcenter, haloghostcenter, self.decimal_precision)

      # Number of cells
      nb_cells += d_cells.shape[0]
    self.logger.testing("Cell Number of cells", np.testing.assert_equal, self.test_tables.nb_cells, nb_cells)

  def test_node_info(self):


    for p in range(self.nb_partitions):
      d_cell_loctoglob = self.domain_tables.d_cell_loctoglob[p]

      d_cells = self.domain_tables.d_cells[p]
      d_nodes = self.domain_tables.d_nodes[p][:, 0:3]
      dglobal_nodes = self.unified_domain.d_nodes[0][:, 0:3]
      d_node_loctoglob = self.domain_tables.d_node_loctoglob[p]
      d_node_cellid = self.domain_tables.d_node_cellid[p]
      d_node_name = self.domain_tables.d_node_name[p]
      d_node_oldname = self.domain_tables.d_node_oldname[p]
      d_node_ghostid = self.domain_tables.d_node_ghostid[p]
      d_node_haloghostid = self.domain_tables.d_node_haloghostid[p]
      d_node_ghostcenter = self.domain_tables.d_node_ghostcenter[p]
      d_node_haloghostcenter = self.domain_tables.d_node_haloghostcenter[p]
      d_node_ghostfaceinfo = self.domain_tables.d_node_ghostfaceinfo[p]
      d_node_haloghostfaceinfo = self.domain_tables.d_node_haloghostfaceinfo[p]
      d_node_halonid = self.domain_tables.d_node_halonid[p]
      d_halo_halosext = self.domain_tables.d_halo_halosext[p]
      d_face_ghostcenter = self.domain_tables.d_face_ghostcenter[p]
      d_cell_haloghostcenter = self.domain_tables.d_cell_haloghostcenter[p]

      for i in range(len(d_cell_loctoglob)):
        g_index = d_cell_loctoglob[i]
        cnb_nodes = d_cells[i][-1] # number of cell nodes

        # Vertices (already checked on cell_vertices)

        # Cellid
        for k in range(cnb_nodes):
          node_cellid = self.test_tables.l_node_cellid[g_index]
          node_cellid = node_cellid[k][0:node_cellid[k][-1]]
          node_cellid = np.sort(node_cellid)

          c_nodeid = d_cells[i][k]
          c_node_cellid = d_node_cellid[c_nodeid][0:d_node_cellid[c_nodeid][-1]]
          c_node_cellid = d_cell_loctoglob[c_node_cellid]
          c_node_cellid = np.sort(c_node_cellid)

          self.logger.testing("Node Cellid", np.testing.assert_equal, node_cellid, c_node_cellid)


        # Loctoglob
        cell_vertices = self.test_tables.cell_vertices[g_index]

        c_cell_nodeid = d_cells[i][0:d_cells[i][-1]]
        c_cell_nodeid = d_node_loctoglob[c_cell_nodeid]
        c_cell_vertices = dglobal_nodes[c_cell_nodeid]
        self.logger.testing("Node Loctoglob", np.testing.assert_almost_equal, cell_vertices, c_cell_vertices, decimal=self.decimal_precision)

        # Halonid
        node_halonid = self.test_tables.node_halonid[g_index]

        c_cell_nodes = d_cells[i][0:d_cells[i][-1]]
        for k in range(cnb_nodes):
          c_node_halonid = d_node_halonid[c_cell_nodes[k]]
          c_node_halonid = c_node_halonid[0:c_node_halonid[-1]]
          c_node_halonid = d_halo_halosext[c_node_halonid][:, 0]
          c_node_halonid = np.sort(c_node_halonid)
          self.logger.testing("Node Halonid", np.testing.assert_equal, node_halonid[k][0:node_halonid[k][-1]], c_node_halonid)

        # # Oldname
        # node_oldname = self.test_tables.g_node_name[g_index]
        # c_cell_nodes = d_cells[i][0:d_cells[i][-1]]
        # c_node_oldname = d_node_oldname[c_cell_nodes]
        # self.logger.testing("Node Oldname", np.testing.assert_equal, node_oldname, c_node_oldname)
        #
        # # # Name
        # cell_nodes = self.test_tables.g_cell_nodeid[g_index]
        # cell_nodes = cell_nodes[0:cell_nodes[-1]]
        # node_name = self.test_tables.l_node_name[cell_nodes]
        # c_cell_nodes = d_cells[i][0:d_cells[i][-1]]
        # c_node_name = d_node_name[c_cell_nodes]
        # self.logger.testing("Node Name", np.testing.assert_equal, node_name, c_node_name)
        #
        #
        # # Node: ghostid, ghostcenter, ghostfaceinfo TODO ghost center gamma
        # # tester.ghost_info => [center_x, center_y, center_z, volume, cell_partition_id, cell_id, cell_face_id(0..6)]
        # # nodes_ghostcenter => [[g_x, g_y, g_z, cell_id, face_old_name, ghostid] * nb_ghost_of_a_node] * nb_node
        # # nodes_ghostfaceinfo => [[face_center_x, face_center_y, face_center_z, face_normal_x, face_normal_y, face_normal_z] * nb_ghost_of_a_node] * nb_nodes
        # c_cell_nodes = d_cells[i][0:d_cells[i][-1]]
        # for k in range(cnb_nodes):
        #   # 6 -> node_ghostid.shape[1]-1
        #   ghostinfo = np.ones(shape=(6, 15), dtype=np.float32) * -1
        #   node_ghostid = self.test_tables.l_node_ghostid[g_index]
        #   node_ghostid = node_ghostid[k][0:node_ghostid[k][-1]]
        #   node_ghostinfo = self.test_tables.ghost_info[node_ghostid]
        #   node_cellid = node_ghostinfo[:, 5].astype(np.int32)
        #   node_faceid = node_ghostinfo[:, 6].astype(np.int32)
        #   nb_ghost = len(node_ghostinfo)
        #
        #   ghostinfo[0:nb_ghost, 0] = node_ghostinfo[:, 0] #g_x
        #   ghostinfo[0:nb_ghost, 1] = node_ghostinfo[:, 1] #g_y
        #   ghostinfo[0:nb_ghost, 2] = node_ghostinfo[:, 2] #g_z
        #   ghostinfo[0:nb_ghost, 3] = node_cellid #cell_id
        #   ghostinfo[0:nb_ghost, 4] = self.test_tables.l_face_name[node_cellid, node_faceid] # face_oldname
        #   ghostinfo[0:nb_ghost, 5] = node_ghostinfo[:, 0] #g_x
        #   ghostinfo[0:nb_ghost, 6] = node_ghostinfo[:, 1] #g_y
        #   ghostinfo[0:nb_ghost, 7] = node_ghostinfo[:, 2] #g_z
        #   # ghostinfo[0:nb_ghost, 8] = node_ghostinfo[:, 3] #vol
        #   ghostinfo[0:nb_ghost, 9] = self.test_tables.face_center[node_cellid, node_faceid][:, 0] # face_center_x
        #   ghostinfo[0:nb_ghost, 10] = self.test_tables.face_center[node_cellid, node_faceid][:, 1] # face_center_y
        #   ghostinfo[0:nb_ghost, 11] = self.test_tables.face_center[node_cellid, node_faceid][:, 2] # face_center_z
        #   ghostinfo[0:nb_ghost, 12] = self.test_tables.face_normal[node_cellid, node_faceid][:, 0] # face_normal_x
        #   ghostinfo[0:nb_ghost, 13] = self.test_tables.face_normal[node_cellid, node_faceid][:, 1] # face_normal_y
        #   ghostinfo[0:nb_ghost, 14] = self.test_tables.face_normal[node_cellid, node_faceid][:, 2] # face_normal_z
        #
        #
        #   ##########################
        #
        #   c_ghostinfo = np.ones(shape=(6, 15), dtype=np.float32) * -1
        #   c_node_ghostcenter = d_node_ghostcenter[c_cell_nodes[k]]
        #   c_node_ghostfaceinfo = d_node_ghostfaceinfo[c_cell_nodes[k]]
        #   c_node_ghostid = d_node_ghostid[c_cell_nodes[k]]
        #   c_node_ghostid = c_node_ghostid[0:c_node_ghostid[-1]]
        #   c_node_cellid = c_node_ghostcenter[:, 3].astype(np.int32)
        #   c_node_cellid = c_node_cellid[c_node_cellid != -1]
        #   c_nb_ghost = len(c_node_ghostid)
        #
        #   c_ghostinfo[0:c_nb_ghost, 0] = c_node_ghostcenter[0:c_nb_ghost, 0] #g_x
        #   c_ghostinfo[0:c_nb_ghost, 1] = c_node_ghostcenter[0:c_nb_ghost, 1] #g_y
        #   c_ghostinfo[0:c_nb_ghost, 2] = c_node_ghostcenter[0:c_nb_ghost, 2] #g_z
        #   c_ghostinfo[0:c_nb_ghost, 3] = d_cell_loctoglob[c_node_cellid] #cell_id
        #   c_ghostinfo[0:c_nb_ghost, 4] = c_node_ghostcenter[0:c_nb_ghost, 4] # face_old_name
        #   c_ghostinfo[0:c_nb_ghost, 5] = d_face_ghostcenter[c_node_ghostid, 0] # g_x from ghostid
        #   c_ghostinfo[0:c_nb_ghost, 6] = d_face_ghostcenter[c_node_ghostid, 1] # g_y from ghostid
        #   c_ghostinfo[0:c_nb_ghost, 7] = d_face_ghostcenter[c_node_ghostid, 2] # g_z from ghostid
        #   # c_ghostinfo[0:c_nb_ghost, 8] = d_face_ghostcenter[c_node_ghostid, 3] # vol from ghostid
        #   c_ghostinfo[0:c_nb_ghost, 9] = c_node_ghostfaceinfo[0:c_nb_ghost, 0] # face_center_x
        #   c_ghostinfo[0:c_nb_ghost, 10] = c_node_ghostfaceinfo[0:c_nb_ghost, 1] # face_center_y
        #   c_ghostinfo[0:c_nb_ghost, 11] = c_node_ghostfaceinfo[0:c_nb_ghost, 2] # face_center_z
        #   c_ghostinfo[0:c_nb_ghost, 12] = c_node_ghostfaceinfo[0:c_nb_ghost, 3] # face_normal_x
        #   c_ghostinfo[0:c_nb_ghost, 13] = c_node_ghostfaceinfo[0:c_nb_ghost, 4] # face_normal_y
        #   c_ghostinfo[0:c_nb_ghost, 14] = c_node_ghostfaceinfo[0:c_nb_ghost, 5] # face_normal_z
        #
        #   ghostinfo = self.sort_float_arr(ghostinfo)
        #   c_ghostinfo = self.sort_float_arr(c_ghostinfo)
        #   self.logger.testing("Node ghostid, ghostcenter and ghostfaceinfo *", np.testing.assert_almost_equal, ghostinfo, c_ghostinfo, decimal=self.decimal_precision)
        # # Node: haloghostid, haloghostcenter, haloghostfaceinfo
        # # The same code as above except that tables are become the halo's tables
        # c_cell_nodes = d_cells[i][0:d_cells[i][-1]]
        # for k in range(cnb_nodes):
        #   # 6 -> node_ghostid.shape[1]-1
        #   haloghostinfo = np.ones(shape=(6, 15), dtype=np.float32) * -1
        #   node_haloghostid = self.test_tables.node_haloghostid[g_index]
        #   node_haloghostid = node_haloghostid[k][0:node_haloghostid[k][-1]]
        #   node_haloghostinfo = self.test_tables.ghost_info[node_haloghostid]
        #
        #   node_cellid = node_haloghostinfo[:, 5].astype(np.int32)
        #   node_faceid = node_haloghostinfo[:, 6].astype(np.int32)
        #   nb_ghost = len(node_haloghostinfo)
        #
        #   haloghostinfo[0:nb_ghost, 0] = node_haloghostinfo[:, 0] #g_x
        #   haloghostinfo[0:nb_ghost, 1] = node_haloghostinfo[:, 1] #g_y
        #   haloghostinfo[0:nb_ghost, 2] = node_haloghostinfo[:, 2] #g_z
        #   haloghostinfo[0:nb_ghost, 3] = self.test_tables.l_face_name[node_cellid, node_faceid] # face_oldname
        #   haloghostinfo[0:nb_ghost, 4] = node_cellid #cell_id
        #   haloghostinfo[0:nb_ghost, 5] = node_haloghostinfo[:, 0] #g_x
        #   haloghostinfo[0:nb_ghost, 6] = node_haloghostinfo[:, 1] #g_y
        #   haloghostinfo[0:nb_ghost, 7] = node_haloghostinfo[:, 2] #g_z
        #   #haloghostinfo[0:nb_ghost, 8] = node_haloghostinfo[:, 3] #vol
        #   haloghostinfo[0:nb_ghost, 9] = self.test_tables.face_center[node_cellid, node_faceid][:, 0] # face_center_x
        #   haloghostinfo[0:nb_ghost, 10] = self.test_tables.face_center[node_cellid, node_faceid][:, 1] # face_center_y
        #   haloghostinfo[0:nb_ghost, 11] = self.test_tables.face_center[node_cellid, node_faceid][:, 2] # face_center_z
        #   haloghostinfo[0:nb_ghost, 12] = self.test_tables.face_normal[node_cellid, node_faceid][:, 0] # face_normal_x
        #   haloghostinfo[0:nb_ghost, 13] = self.test_tables.face_normal[node_cellid, node_faceid][:, 1] # face_normal_y
        #   haloghostinfo[0:nb_ghost, 14] = self.test_tables.face_normal[node_cellid, node_faceid][:, 2] # face_normal_z
        #
        #
        #   ##########################
        #
        #   c_haloghostinfo = np.ones(shape=(6, 15), dtype=np.float32) * -1
        #   c_node_haloghostcenter = d_node_haloghostcenter[c_cell_nodes[k]]
        #   c_node_haloghostfaceinfo = d_node_haloghostfaceinfo[c_cell_nodes[k]]
        #   c_node_haloghostid = d_node_haloghostid[c_cell_nodes[k]]
        #   c_node_haloghostid = c_node_haloghostid[0:c_node_haloghostid[-1]]
        #   c_node_cellid = c_node_haloghostcenter[:, 3].astype(np.int32)
        #   c_node_cellid = c_node_cellid[c_node_cellid != -1]
        #   c_nb_ghost = len(c_node_haloghostid)
        #
        #   c_haloghostinfo[0:c_nb_ghost, 0] = c_node_haloghostcenter[0:c_nb_ghost, 0] #g_x
        #   c_haloghostinfo[0:c_nb_ghost, 1] = c_node_haloghostcenter[0:c_nb_ghost, 1] #g_y
        #   c_haloghostinfo[0:c_nb_ghost, 2] = c_node_haloghostcenter[0:c_nb_ghost, 2] #g_z
        #   c_haloghostinfo[0:c_nb_ghost, 3] = c_node_haloghostcenter[0:c_nb_ghost, 4] # face_old_name
        #   c_haloghostinfo[0:c_nb_ghost, 4] = d_halo_halosext[c_node_cellid][:, 0] #cell_id
        #
        #   c_haloghostinfo[0:c_nb_ghost, 5] = d_cell_haloghostcenter[c_node_haloghostid][:, 0] # g_x from haloghostid
        #   c_haloghostinfo[0:c_nb_ghost, 6] = d_cell_haloghostcenter[c_node_haloghostid][:, 1] # g_y from haloghostid
        #   c_haloghostinfo[0:c_nb_ghost, 7] = d_cell_haloghostcenter[c_node_haloghostid][:, 2] # g_z from haloghostid
        #   #c_haloghostinfo[0:c_nb_ghost, 8] = d_face_ghostcenter[c_node_haloghostid, 3] # no need
        #   c_haloghostinfo[0:c_nb_ghost, 9] = c_node_haloghostfaceinfo[0:c_nb_ghost, 0] # face_center_x
        #   c_haloghostinfo[0:c_nb_ghost, 10] = c_node_haloghostfaceinfo[0:c_nb_ghost, 1] # face_center_y
        #   c_haloghostinfo[0:c_nb_ghost, 11] = c_node_haloghostfaceinfo[0:c_nb_ghost, 2] # face_center_z
        #   c_haloghostinfo[0:c_nb_ghost, 12] = c_node_haloghostfaceinfo[0:c_nb_ghost, 3] # face_normal_x
        #   c_haloghostinfo[0:c_nb_ghost, 13] = c_node_haloghostfaceinfo[0:c_nb_ghost, 4] # face_normal_y
        #   c_haloghostinfo[0:c_nb_ghost, 14] = c_node_haloghostfaceinfo[0:c_nb_ghost, 5] # face_normal_z
        #
        #   haloghostinfo = self.sort_float_arr(haloghostinfo)
        #   c_haloghostinfo = self.sort_float_arr(c_haloghostinfo)
        #   self.logger.testing("Node haloghostid, haloghostcenter and haloghostfaceinfo", np.testing.assert_almost_equal, haloghostinfo, c_haloghostinfo, decimal=self.decimal_precision)

    # Node: number of nodes
    a = np.concatenate(self.domain_tables.d_nodes)
    a = np.round(a[:, 0:3], decimals=2)
    a = np.unique(a, axis=0)
    self.logger.testing("Node Number of nodes", np.testing.assert_equal, self.test_tables.nb_nodes, a.shape[0])

  def test_face_info(self):

    for p in range(self.nb_partitions):
      d_cell_loctoglob = self.domain_tables.d_cell_loctoglob[p]

      d_cells = self.domain_tables.d_cells[p]
      d_nodes = self.domain_tables.d_nodes[p][:, 0:3]
      d_faces = self.domain_tables.d_faces[p][:, 0:3] #triangle
      d_cell_faces = self.domain_tables.d_cell_faces[p]
      d_cell_nf = self.domain_tables.d_cell_nf[p][:, :, 0:6] #six faces

      d_face_measure = self.domain_tables.d_face_measure[p]
      d_face_center = self.domain_tables.d_face_center[p][:, 0:3]
      d_face_normal = self.domain_tables.d_face_normal[p][:, 0:3]
      d_face_ghostcenter = self.domain_tables.d_face_ghostcenter[p]
      d_face_name = self.domain_tables.d_face_name[p]
      d_face_oldname = self.domain_tables.d_face_oldname[p]
      d_face_cellid = self.domain_tables.d_face_cellid[p]

      for i in range(len(d_cell_loctoglob)):
        g_index = d_cell_loctoglob[i]

        c_faces = d_cell_faces[i][0:d_cell_faces[i][-1]]
        c_faces_nodes = d_faces[c_faces]
        c_nodes = d_cells[i][0:d_cells[i][-1]]
        c_faces_nodes, sorted_indexes = self._sort_cell_faces(c_nodes, c_faces_nodes)
        c_faces = c_faces[sorted_indexes]

        # Vertices
        faces_vertices = self.test_tables.faces_vertices[g_index]
        c_faces_vertices = d_nodes[c_faces_nodes] #compare cell faces point vertices (cell.faceid)
        self.logger.testing("Face Vertices", np.testing.assert_almost_equal, faces_vertices, c_faces_vertices, self.decimal_precision)

        # Measure
        faces_measure = self.test_tables.faces_measure[g_index]
        c_faces_measure = d_face_measure[c_faces]
        self.logger.testing("Face Measure", np.testing.assert_almost_equal, faces_measure, c_faces_measure, self.decimal_precision)

        # Face center
        faces_center = self.test_tables.face_center[g_index]
        c_faces_center = d_face_center[c_faces]
        self.logger.testing("Face Center", np.testing.assert_almost_equal, faces_center, c_faces_center, self.decimal_precision)

        # Name
        faces_name = self.test_tables.l_face_name[g_index]
        c_faces_name = d_face_name[c_faces]
        self.logger.testing("Face Name", np.testing.assert_equal, faces_name, c_faces_name)

        # Oldname
        faces_oldname = self.test_tables.l_face_name[g_index]
        c_faces_oldname = d_face_oldname[c_faces]
        self.logger.testing("Face Oldname", np.testing.assert_equal, faces_oldname, c_faces_oldname)

        #! Normal (Only abs)
        faces_normal = self.test_tables.face_normal[g_index]
        faces_normal = np.abs(faces_normal)
        c_faces_normal = np.abs(d_face_normal[c_faces])
        self.logger.testing("Face Normal(Only abs)", np.testing.assert_almost_equal, faces_normal, c_faces_normal, self.decimal_precision)

        # CellId
        faces_cellid = self.test_tables.l_face_cellid[g_index]
        faces_cellid = np.sort(faces_cellid)
        c_faces_cellid = d_face_cellid[c_faces]
        c_faces_cellid[c_faces_cellid < 0] = -1
        tmp = d_cell_loctoglob[c_faces_cellid].copy()
        tmp[c_faces_cellid == -1] = -1
        c_faces_cellid = np.sort(tmp)
        self.logger.testing("Face CellId", np.testing.assert_equal, faces_cellid, c_faces_cellid)

        # Ghostcenter TODO Changed Need to check gamma ddm_utils2d.py => face_info_2d => line 187
        faces_ghostcenter = self.test_tables.face_ghostcenter[g_index][:, 0:3]
        c_faces_ghostcenter = d_face_ghostcenter[c_faces][:, 0:3]
        self.logger.testing("Face Ghostcenter", np.testing.assert_almost_equal, faces_ghostcenter, c_faces_ghostcenter, self.decimal_precision)

        # # Cell face normal (cells.cell_nf) # TODO cells.cell_nf not used on 3D
        # cell_nf = self.test_tables.cell_nf[g_index]
        # c_cell_nf = d_cell_nf[i]
        # self.logger.testing("Face Cell face normal", np.testing.assert_almost_equal, cell_nf, c_cell_nf, self.decimal_precision)

    # Face: number of faces
    a = np.concatenate(self.domain_tables.d_face_center)
    a = np.round(a[:, 0:3], decimals=2)
    a = np.unique(a, axis=0)
    self.logger.testing("Face Number of faces", np.testing.assert_equal, self.test_tables.nb_faces, a.shape[0])

  def test_halo_info(self):
    if (self.nb_partitions <= 1): # don't test on nb_partition = 1
      return

    for p in range(self.nb_partitions):
      d_cells = self.domain_tables.d_cells[p]
      d_halo_halosext = self.domain_tables.d_halo_halosext[p]
      d_halo_halosint = self.domain_tables.d_halo_halosint[p]
      d_halo_neigh = self.domain_tables.d_halo_neigh[p]
      d_halo_centvol = self.domain_tables.d_halo_centvol[p]
      d_halo_sizehaloghost = self.domain_tables.d_halo_sizehaloghost

      # Halo: Halosext Already tested on [test_face_info, test_cell_info, test_node_info]
      # Test Halosext node ids
      d_halosext = d_halo_halosext[:, 0]
      for i in range(len(d_halosext)):
        halosext_nodes = self.test_tables.g_cell_nodeid[d_halosext[i]]
        halosext_nodes = halosext_nodes[0:halosext_nodes[-1]]
        d_halosext_nodes = d_halo_halosext[i, 1:d_halo_halosext[i][-1]]

        self.logger.testing("Halo Halosext", np.testing.assert_equal, halosext_nodes, d_halosext_nodes)

      # Halo: Halosint
      halo_halosint = self.test_tables.halo_halosint[p]
      halo_halosint = halo_halosint[halo_halosint != -1]
      c_halo_halosint = np.unique(d_halo_halosint)
      self.logger.testing("Halo Halosint", np.testing.assert_equal, halo_halosint, c_halo_halosint)

      # Halo: neigh
      halo_neigh = self.test_tables.halo_neigh
      self.logger.testing("Halo neigh", np.testing.assert_equal, halo_neigh[d_halo_neigh[0], p], d_halo_neigh[1])

      # Halo: centvol
      halosext_ids = d_halo_halosext[:, 0]
      halosext_center = self.test_tables.cell_center[halosext_ids]
      halosext_vol = self.test_tables.cell_area[halosext_ids]

      self.logger.testing("Halo center", np.testing.assert_almost_equal, halosext_center, d_halo_centvol[:, 0:3], decimal=self.decimal_precision)
      self.logger.testing("Halo vol", np.testing.assert_almost_equal, halosext_vol, d_halo_centvol[:, 3], decimal=self.decimal_precision)

      # Halo : sizehaloghost
      sizehaloghost = self.test_tables.halo_sizehaloghost
      c_sizehaloghost = d_halo_sizehaloghost

      self.logger.testing("Halo sizehaloghost *", np.testing.assert_equal, sizehaloghost, c_sizehaloghost)




################
## Usage
################

# float_precision = 'float32'
# d_cell_loctoglob = domain_tables.d_cell_loctoglob
# g_cell_nodeid = unified_domain.d_cell_nodeid[0]
# test_tables = TestTablesTriangles2D(float_precision, d_cell_loctoglob, g_cell_nodeid)
# test_tables.init()

# checker = Checker2D(decimal_precision=4, domain_tables=domain_tables, unified_domain=unified_domain, test_tables=test_tables)
# checker.test_cell_info()
# checker.test_face_info()
# checker.test_node_info()
# checker.test_halo_info()
# checker.summary()