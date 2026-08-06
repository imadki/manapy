from manapy.domain import Domain
from manapy.backends import ManapyArray
import numpy as np
from manapy.testing.test_domain_helper import sort_float_arr
from manapy.testing.ReferenceTables import ReferenceTables

"""
local_domains and reference_domains fixture are defined on conftest.py thery are loaded automatically from the subclasses using pytest.
"""


class _NumpyDomainView:
  """Read-only host view of a domain for NumPy-based assertions."""

  _table_names = {"cells", "faces", "nodes", "ghost", "halos"}

  def __init__(self, obj):
    self._obj = obj

  def __getattr__(self, name):
    value = getattr(self._obj, name)
    if isinstance(value, ManapyArray):
      return value.cpu_r()
    if name in self._table_names:
      return _NumpyDomainView(value)
    return value


class BaseTestDomain:
  def setup_method(self):
    self.decimal = 5

  def test_cell_vertices(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        node_ids = ld.cells.nodeid[i]
        node_ids = node_ids[0:node_ids[-1]]
        d_vertices = ld.nodes.vertex[node_ids]

        g_id = ld.cells.loctoglob[i]
        node_ids = reference_domain.cells[g_id]
        node_ids = node_ids[0:node_ids[-1]]
        r_vertices = reference_domain.nodes[node_ids]

        np.testing.assert_almost_equal(d_vertices, r_vertices, decimal=5)

  def test_cell_center(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        d_cell_center = ld.cells.center[i]

        g_id = ld.cells.loctoglob[i]
        r_cell_center = reference_domain.cell_center[g_id]

        np.testing.assert_almost_equal(d_cell_center, r_cell_center, decimal=self.decimal)

  def test_cell_area(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        d_cell_area = ld.cells.volume[i]

        g_id = ld.cells.loctoglob[i]
        r_cell_area = reference_domain.cell_area[g_id]

        np.testing.assert_almost_equal(d_cell_area, r_cell_area, decimal=self.decimal)

  def test_cell_neighbor_by_face(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_cellfid = ld.cells.cellfid[i]
        cell_cellfid = cell_cellfid[0:cell_cellfid[-1]] # local_cellid
        cell_cellfid = ld.cells.loctoglob[cell_cellfid] # global_cellid
        d_cell_cellfid = np.sort(cell_cellfid) # sort cellid

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_cellfid = reference_domain.locals[part].cell_cellfid[l_id]
        cell_cellfid = cell_cellfid[0:cell_cellfid[-1]]  # global_cellid
        r_cell_cellfid = np.sort(cell_cellfid)  # sort cellid

        np.testing.assert_equal(d_cell_cellfid, r_cell_cellfid)

  def test_cell_neighbor_by_node(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_cellnid = ld.cells.cellnid[i]
        cell_cellnid = cell_cellnid[0:cell_cellnid[-1]] # local_cellid
        cell_cellnid = ld.cells.loctoglob[cell_cellnid] # global_cellid
        d_cell_cellnid = np.sort(cell_cellnid) # sort cellid

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_cellnid = reference_domain.locals[part].cell_cellnid[l_id]
        cell_cellnid = cell_cellnid[0:cell_cellnid[-1]]  # global_cellid
        r_cell_cellnid = np.sort(cell_cellnid)  # sort cellid

        np.testing.assert_equal(d_cell_cellnid, r_cell_cellnid)

  def test_cell_halo_by_face(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        cell_halofid =ld.faces.halofid[cell_faces]
        cell_halofid = cell_halofid[cell_halofid != -1] # get neighbor halos by face
        cell_halofid = ld.halos.halosext[cell_halofid][:, 0] # get cell global ID
        d_cell_halofid = np.sort(cell_halofid)

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_halofid = reference_domain.locals[part].cell_halofid[l_id]
        cell_halofid = cell_halofid[0:cell_halofid[-1]]
        r_cell_halofid = np.sort(cell_halofid)

        np.testing.assert_equal(d_cell_halofid, r_cell_halofid)

  def test_cell_halo_by_node(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_halonid = ld.cells.halonid[i]
        cell_halonid = cell_halonid[0:cell_halonid[-1]] # get neighbor halos by node
        cell_halonid = ld.halos.halosext[cell_halonid][:, 0] # get cell global ID
        d_cell_halonid = np.sort(cell_halonid)

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_halonid = reference_domain.locals[part].cell_halonid[l_id]
        cell_halonid = cell_halonid[0:cell_halonid[-1]]
        r_cell_halonid = np.sort(cell_halonid)

        np.testing.assert_equal(d_cell_halonid, r_cell_halonid)

  def test_cell_ghostid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_ghostnid = ld.cells.ghostnid[i]
        cell_ghostnid = cell_ghostnid[0:cell_ghostnid[-1]]
        cell_ghost_center = ld.ghost.info_flt[cell_ghostnid]
        cell_ghost_center = cell_ghost_center[:, 0:dim]
        d_cell_ghost_center = sort_float_arr(dim, cell_ghost_center)[0]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_ghostnid = reference_domain.locals[part].cell_ghostnid[l_id]
        cell_ghostnid = cell_ghostnid[0:cell_ghostnid[-1]]
        cell_ghost_center = reference_domain.ghost_info_flt[cell_ghostnid][:, [0, 1, 2]] # (center only)
        cell_ghost_center = cell_ghost_center[:, 0:dim]
        r_cell_ghost_center = sort_float_arr(dim, cell_ghost_center)[0]

        for j in range(len(d_cell_ghost_center)):
          np.testing.assert_almost_equal(d_cell_ghost_center[j], r_cell_ghost_center[j], decimal=self.decimal)
        np.testing.assert_equal(len(d_cell_ghost_center), len(r_cell_ghost_center))

  def test_cell_halo_ghostid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_haloghostnid = ld.cells.haloghostnid[i]
        cell_haloghostnid = cell_haloghostnid[0:cell_haloghostnid[-1]]
        cell_haloghost_center = ld.ghost.ext_info_flt[cell_haloghostnid]
        cell_haloghost_center = cell_haloghost_center[:, 0:dim]
        d_cell_haloghost_center = sort_float_arr(dim, cell_haloghost_center)[0]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        cell_haloghostnid = reference_domain.locals[part].cell_haloghostnid[l_id]
        cell_haloghostnid = cell_haloghostnid[0:cell_haloghostnid[-1]]
        cell_haloghost_center = reference_domain.ghost_info_flt[cell_haloghostnid][:, [0, 1, 2]] # (center only)
        cell_haloghost_center = cell_haloghost_center[:, 0:dim]
        r_cell_haloghost_center = sort_float_arr(dim, cell_haloghost_center)[0]

        for j in range(len(d_cell_haloghost_center)):
          np.testing.assert_almost_equal(d_cell_haloghost_center[j], r_cell_haloghost_center[j], decimal=self.decimal)
        np.testing.assert_equal(len(d_cell_haloghost_center), len(r_cell_haloghost_center))

  def test_nb_cells(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    nb_cells = 0
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      nb_cells += len(ld.cells.nodeid)

    np.testing.assert_equal(nb_cells, reference_domain.nb_cells)

  # #################################################

  def test_node_cellid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_nodes = reference_domain.locals[part].cells[l_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]

        assert len(r_cell_nodes) == len(cell_nodes)

        for k in range(len(cell_nodes)):
          node_id = cell_nodes[k]
          node_cellid = ld.nodes.cellid[node_id]
          node_cellid = node_cellid[0:node_cellid[-1]]
          node_cellid = ld.cells.loctoglob[node_cellid]
          d_node_cellid = np.sort(node_cellid)

          r_node_id = r_cell_nodes[k]
          r_node_cellid = reference_domain.locals[part].node_cellid[r_node_id]
          r_node_cellid = r_node_cellid[0:r_node_cellid[-1]]
          r_node_cellid = np.sort(r_node_cellid)

          np.testing.assert_equal(d_node_cellid, r_node_cellid)

  def test_node_loctoglob(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]
        c_node_loctoglob = ld.nodes.loctoglob[cell_nodes]

        g_id = ld.cells.loctoglob[i]
        g_cell_nodes = reference_domain.meshio_cells[g_id]
        r_cell_nodes = g_cell_nodes[0:g_cell_nodes[-1]]

        print(g_id, c_node_loctoglob)
        np.testing.assert_equal(c_node_loctoglob, r_cell_nodes)

  def test_node_halonid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_nodes = reference_domain.locals[part].cells[l_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]

        assert len(r_cell_nodes) == len(cell_nodes)

        for k in range(len(cell_nodes)):
          node_id = cell_nodes[k]
          node_halonid = ld.nodes.halonid[node_id]
          node_halonid = node_halonid[0:node_halonid[-1]]
          node_halonid = ld.halos.halosext[node_halonid][:, 0]
          d_node_halonid = np.sort(node_halonid)

          r_node_id = r_cell_nodes[k]
          r_node_halonid = reference_domain.locals[part].node_halonid[r_node_id]
          r_node_halonid = r_node_halonid[0:r_node_halonid[-1]]
          r_node_halonid = np.sort(r_node_halonid)

          np.testing.assert_equal(d_node_halonid, r_node_halonid)

  def test_node_oldname(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]
        d_node_oldname = ld.nodes.oldname[cell_nodes]

        g_id = ld.cells.loctoglob[i]
        r_cell_nodes = reference_domain.cells[g_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]
        r_node_oldname = reference_domain.node_oldname[r_cell_nodes]

        np.testing.assert_equal(d_node_oldname, r_node_oldname)

  def test_node_name(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]
        d_node_name = ld.nodes.name[cell_nodes]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_nodes = reference_domain.locals[part].cells[l_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]
        r_node_name = reference_domain.locals[part].node_name[r_cell_nodes]

        np.testing.assert_equal(d_node_name, r_node_name)

  def test_node_ghostnid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_nodes = reference_domain.locals[part].cells[l_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]

        assert len(r_cell_nodes) == len(cell_nodes)

        for k in range(len(cell_nodes)):
          node_id = cell_nodes[k]
          node_ghostnid = ld.nodes.ghostid[node_id]
          node_ghostnid = node_ghostnid[0:node_ghostnid[-1]]
          d_node_ghost_center = ld.ghost.info_flt[node_ghostnid]
          d_node_ghost_center = d_node_ghost_center[:, 0:dim]
          d_node_ghost_center = sort_float_arr(dim, d_node_ghost_center)[0]

          r_node_id = r_cell_nodes[k]
          r_node_ghostnid = reference_domain.locals[part].node_ghostnid[r_node_id]
          r_node_ghostnid = r_node_ghostnid[0:r_node_ghostnid[-1]]
          r_node_ghost_center = reference_domain.ghost_info_flt[r_node_ghostnid][:, [0, 1, 2]]
          r_node_ghost_center = r_node_ghost_center[:, 0:dim]
          r_node_ghost_center = sort_float_arr(dim, r_node_ghost_center)[0]

          for j in range(len(d_node_ghost_center)):
            np.testing.assert_almost_equal(d_node_ghost_center[j], r_node_ghost_center[j], decimal=self.decimal)
          np.testing.assert_equal(len(d_node_ghost_center), len(r_node_ghost_center))

  def test_node_haloghostnid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_nodes = ld.cells.nodeid[i]
        cell_nodes = cell_nodes[0:cell_nodes[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_nodes = reference_domain.locals[part].cells[l_id]
        r_cell_nodes = r_cell_nodes[0:r_cell_nodes[-1]]

        assert len(r_cell_nodes) == len(cell_nodes)

        for k in range(len(cell_nodes)):
          node_id = cell_nodes[k]
          node_haloghostnid = ld.nodes.haloghostid[node_id]
          node_haloghostnid = node_haloghostnid[0:node_haloghostnid[-1]]
          d_node_haloghost_center = ld.ghost.ext_info_flt[node_haloghostnid]
          d_node_haloghost_center = d_node_haloghost_center[:, 0:dim]
          d_node_haloghost_center = sort_float_arr(dim, d_node_haloghost_center)[0]

          r_node_id = r_cell_nodes[k]
          r_node_haloghostnid = reference_domain.locals[part].node_haloghostnid[r_node_id]
          r_node_haloghostnid = r_node_haloghostnid[0:r_node_haloghostnid[-1]]
          r_node_haloghost_center = reference_domain.ghost_info_flt[r_node_haloghostnid][:, [0, 1, 2]]
          r_node_haloghost_center = r_node_haloghost_center[:, 0:dim]
          r_node_haloghost_center = sort_float_arr(dim, r_node_haloghost_center)[0]

          for j in range(len(d_node_haloghost_center)):
            np.testing.assert_almost_equal(d_node_haloghost_center[j], r_node_haloghost_center[j], decimal=self.decimal)
          np.testing.assert_equal(len(d_node_haloghost_center), len(r_node_haloghost_center))

  def test_nb_nodes(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    all_nodes = []
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      all_nodes.append(ld.nodes.vertex[:, 0:3])

    a = np.concatenate(all_nodes)
    a = np.round(a[:, 0:3], decimals=2)
    a = np.unique(a, axis=0)

    np.testing.assert_equal(a.shape[0], reference_domain.nb_nodes)

  # #################################################

  def test_face_vertices(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        cell_faces = cell_faces[d_order]
        r_cell_faces = r_cell_faces[r_order]

        for j in range(len(cell_faces)):
          face_id = cell_faces[j]
          d_face_nodes = ld.faces.nodeid[face_id]
          d_face_nodes = d_face_nodes[0:d_face_nodes[-1]]
          d_face_vertices = ld.nodes.vertex[d_face_nodes]
          d_face_vertices = sort_float_arr(dim, d_face_vertices)[0]

          r_face_id = r_cell_faces[j]
          r_face_nodes = reference_domain.faces[r_face_id]
          r_face_nodes = r_face_nodes[0:r_face_nodes[-1]]
          r_face_vertices = reference_domain.nodes[r_face_nodes]
          r_face_vertices = sort_float_arr(dim, r_face_vertices)[0]

          np.testing.assert_almost_equal(d_face_vertices, r_face_vertices, decimal=self.decimal)

  def test_face_measure(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        d_face_measure = ld.faces.mesure[cell_faces]

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]
        r_face_measure = reference_domain.face_measure[r_cell_faces]

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        np.testing.assert_almost_equal(d_face_measure[d_order], r_face_measure[r_order], decimal=self.decimal)

  def test_face_center(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        d_face_center = ld.faces.center[cell_faces]

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]
        r_face_center = reference_domain.face_center[r_cell_faces]

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        np.testing.assert_almost_equal(d_face_center[d_order], r_face_center[r_order], decimal=self.decimal)

  def test_face_name(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        d_face_name = ld.faces.name[cell_faces]

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]
        r_face_name = reference_domain.face_name[r_cell_faces]

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        np.testing.assert_equal(d_face_name[d_order], r_face_name[r_order])

  def test_face_oldname(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        d_face_oldname = ld.faces.oldname[cell_faces].copy()
        d_face_oldname[d_face_oldname == 10] = 0

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]
        r_face_oldname = reference_domain.face_oldname[r_cell_faces].copy()
        r_face_oldname[r_face_oldname == 10] = 0

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        np.testing.assert_equal(d_face_oldname[d_order], r_face_oldname[r_order])

  def test_face_normal(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]
        d_face_normal = np.abs(ld.faces.normal[cell_faces])

        g_id = ld.cells.loctoglob[i]
        r_cell_faces = reference_domain.cell_faceid[g_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]
        r_face_normal = np.abs(reference_domain.face_normal[r_cell_faces])

        # order the faces using center coordinate
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[r_cell_faces])

        np.testing.assert_almost_equal(d_face_normal[d_order], r_face_normal[r_order], decimal=self.decimal)

  def test_face_cellid(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_faces = reference_domain.locals[part].cell_faceid[l_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]

        # order the faces using center coordinate
        y = reference_domain.cell_faceid[g_id]
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[y[0:y[-1]]])

        cell_faces = cell_faces[d_order]
        r_cell_faces = r_cell_faces[r_order]

        d_face_cellid = np.copy(ld.faces.cellid[cell_faces])
        r_face_cellid = reference_domain.locals[part].face_cellid[r_cell_faces]

        for i in range(len(d_face_cellid)):
          local_cell_id = d_face_cellid[i]
          if local_cell_id[1] != -1:
            d_face_cellid[i, 1] = ld.cells.loctoglob[local_cell_id[1]]
          if local_cell_id[0] != -1:
            d_face_cellid[i, 0] = ld.cells.loctoglob[local_cell_id[0]]

        d_face_cellid = np.sort(d_face_cellid)
        r_face_cellid = np.sort(r_face_cellid)
        np.testing.assert_equal(d_face_cellid, r_face_cellid)

  def test_face_ghostcenter(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        cell_faces = ld.cells.faceid[i]
        cell_faces = cell_faces[0:cell_faces[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_faces = reference_domain.locals[part].cell_faceid[l_id]
        r_cell_faces = r_cell_faces[0:r_cell_faces[-1]]

        # order the faces using center coordinate
        y = reference_domain.cell_faceid[g_id]
        _, d_order = sort_float_arr(ld.dim, ld.faces.center[cell_faces])
        _, r_order = sort_float_arr(ld.dim, reference_domain.face_center[y[0:y[-1]]])

        cell_faces = cell_faces[d_order]
        r_cell_faces = r_cell_faces[r_order]

        for k in range(len(cell_faces)):
          d_ghost_id = ld.faces.ghost_id[cell_faces[k]]
          r_ghost_id = reference_domain.locals[part].face_ghostid[r_cell_faces[k]]
          if d_ghost_id != -1:
            assert r_ghost_id != -1
            d_ghost_center = ld.ghost.info_flt[d_ghost_id, [0, 1, 2]]
            d_ghost_center = d_ghost_center[0:dim]

            r_ghost_center = reference_domain.ghost_info_flt[r_ghost_id, [0, 1, 2]]
            r_ghost_center = r_ghost_center[0:dim]

            np.testing.assert_almost_equal(d_ghost_center, r_ghost_center, decimal=self.decimal)
          else:
            assert r_ghost_id == -1



  def test_nb_faces(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    all_faces = []
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      all_faces.append(ld.faces.center[:, 0:3])

    a = np.concatenate(all_faces)
    a = np.round(a[:, 0:3], decimals=2)
    a = np.unique(a, axis=0)

    np.testing.assert_equal(a.shape[0], reference_domain.nb_faces)


  # #################################################

  def test_ghost_info(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(ld.cells.nodeid.shape[0]):
        d_cell_ghostnid = ld.cells.ghostnid[i]
        d_cell_ghostnid = d_cell_ghostnid[0:d_cell_ghostnid[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_ghostnid = reference_domain.locals[part].cell_ghostnid[l_id]
        r_cell_ghostnid = r_cell_ghostnid[0:r_cell_ghostnid[-1]]

        d_ghost_info_flt = ld.ghost.info_flt[d_cell_ghostnid]
        d_ghost_info_int = ld.ghost.info_int[d_cell_ghostnid]
        r_ghost_info_flt = reference_domain.ghost_info_flt[r_cell_ghostnid]
        r_ghost_info_int = reference_domain.ghost_info_int[r_cell_ghostnid]

        _, order = sort_float_arr(dim, d_ghost_info_flt)
        d_ghost_info_flt = d_ghost_info_flt[order]
        d_ghost_info_int = d_ghost_info_int[order]

        _, order = sort_float_arr(dim, r_ghost_info_flt)
        r_ghost_info_flt = r_ghost_info_flt[order]
        r_ghost_info_int = r_ghost_info_int[order]

        assert d_ghost_info_flt.shape[0] == r_ghost_info_flt.shape[0]
        assert d_ghost_info_flt.shape[0] == d_ghost_info_int.shape[0]
        for k in range(d_ghost_info_flt.shape[0]):
          # ghost_center
          np.testing.assert_almost_equal(d_ghost_info_flt[k][0:dim], r_ghost_info_flt[k][0:dim], decimal=self.decimal)

          # gamma not tested here

          # face_center
          np.testing.assert_almost_equal(d_ghost_info_flt[k][4:4+dim], r_ghost_info_flt[k][4:4+dim], decimal=self.decimal)

          # face_normal (only abs)
          np.testing.assert_almost_equal(np.abs(d_ghost_info_flt[k][7:7+dim]), np.abs(r_ghost_info_flt[k][7:7+dim]), decimal=self.decimal)

          # cell_id and cell_global_id
          cell_gid = ld.cells.loctoglob[d_ghost_info_int[k][0]]
          np.testing.assert_equal(cell_gid, r_ghost_info_int[k][0])
          np.testing.assert_equal(cell_gid, d_ghost_info_int[k][3])

          # face_id
          d_face_id = ld.cells.faceid[d_ghost_info_int[k, 0], d_ghost_info_int[k, 1]]
          d_face_center = ld.faces.center[d_face_id]
          r_face_id = reference_domain.cell_faceid[r_ghost_info_int[k, 0]][
            r_ghost_info_int[k, 1]]  # find face_id using cell_id and face index inside the cell
          r_face_center = reference_domain.face_center[r_face_id]
          np.testing.assert_almost_equal(d_face_center, r_face_center, decimal=self.decimal)
          assert d_face_id == d_ghost_info_int[k, 4]

          # face_oldname
          np.testing.assert_equal(d_ghost_info_int[k][2], r_ghost_info_int[k][2])


  def test_halo_ghost_info(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])
      dim = ld.dim

      for i in range(len(ld.cells.nodeid)):
        d_cell_haloghostnid = ld.cells.haloghostnid[i]
        d_cell_haloghostnid = d_cell_haloghostnid[0:d_cell_haloghostnid[-1]]

        g_id = ld.cells.loctoglob[i]
        l_id = reference_domain.locals[part].map_cells[g_id]
        r_cell_haloghostnid = reference_domain.locals[part].cell_haloghostnid[l_id]
        r_cell_haloghostnid = r_cell_haloghostnid[0:r_cell_haloghostnid[-1]]

        d_haloghost_info_flt = ld.ghost.ext_info_flt[d_cell_haloghostnid]
        d_haloghost_info_int = ld.ghost.ext_info_int[d_cell_haloghostnid]
        r_haloghost_info_flt = reference_domain.ghost_info_flt[r_cell_haloghostnid]
        r_haloghost_info_int = reference_domain.ghost_info_int[r_cell_haloghostnid]

        _, order = sort_float_arr(dim, d_haloghost_info_flt)
        d_haloghost_info_flt = d_haloghost_info_flt[order]
        d_haloghost_info_int = d_haloghost_info_int[order]

        _, order = sort_float_arr(dim, r_haloghost_info_flt)
        r_haloghost_info_flt = r_haloghost_info_flt[order]
        r_haloghost_info_int = r_haloghost_info_int[order]

        assert d_haloghost_info_flt.shape[0] == r_haloghost_info_flt.shape[0]
        assert d_haloghost_info_flt.shape[0] == d_haloghost_info_int.shape[0]
        for k in range(d_haloghost_info_flt.shape[0]):
          # ghost_center
          np.testing.assert_almost_equal(d_haloghost_info_flt[k][0:dim], r_haloghost_info_flt[k][0:dim], decimal=self.decimal)

          # gamma not tested here

          # face_center
          np.testing.assert_almost_equal(d_haloghost_info_flt[k][4:4+dim], r_haloghost_info_flt[k][4:4+dim], decimal=self.decimal)

          # face_normal (only abs)
          np.testing.assert_almost_equal(np.abs(d_haloghost_info_flt[k][7:7+dim]), np.abs(r_haloghost_info_flt[k][7:7+dim]), decimal=self.decimal)

          # haloext and cell_global_id
          ghost_gid = r_haloghost_info_int[k, 3]
          haloext_gid = ld.halos.halosext[d_haloghost_info_int[k][0]][0]
          np.testing.assert_equal(ghost_gid, haloext_gid)
          np.testing.assert_equal(ghost_gid, d_haloghost_info_int[k][2])

          # face_oldname
          np.testing.assert_equal(d_haloghost_info_int[k][1], r_haloghost_info_int[k][2])

  # #################################################

  def test_halos_halosext(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      for i in range(len(ld.halos.halosext)):
        halo_id = ld.halos.halosext[i, 0] # halo cell Global ID
        halo_nodes = ld.halos.halosext[i]
        d_halo_nodes = halo_nodes[0:halo_nodes[-1]][1:]

        halo_nodes = reference_domain.meshio_cells[halo_id]
        r_halo_nodes = halo_nodes[0:halo_nodes[-1]]

        np.testing.assert_equal(d_halo_nodes, r_halo_nodes)

  def test_halos_halosint(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    # Halo_neigh, Halosint
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      start = 0
      for neigh in range(ld.halos.neigh.shape[1]):
        neigh_part = ld.halos.neigh[0, neigh]
        nb_haloint = ld.halos.neigh[1, neigh]

        d_haloint = ld.halos.halosint[start:start+nb_haloint]
        d_haloint = np.sort(ld.cells.loctoglob[d_haloint])

        t_haloint = reference_domain.locals[part].halos_halosint
        r_haloint = np.sort(t_haloint.get(neigh_part, np.array([])))
        start += nb_haloint
        np.testing.assert_equal(d_haloint, r_haloint)

  def test_halo_centvol(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      halosext_ids = ld.halos.halosext[:, 0]
      r_halosext_center = reference_domain.cell_center[halosext_ids]
      r_halosext_vol = reference_domain.cell_area[halosext_ids]

      d_halo_centvol_center = ld.halos.centvol[:, 0:3]
      d_halo_centvol_vol = ld.halos.centvol[:, 3]

      np.testing.assert_almost_equal(d_halo_centvol_center, r_halosext_center, decimal=self.decimal)
      np.testing.assert_almost_equal(d_halo_centvol_vol, r_halosext_vol, decimal=self.decimal)

  def test_halo_sizehaloghost(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      arr = []
      r_node_haloghostnid = reference_domain.locals[part].node_haloghostnid
      for i in range(r_node_haloghostnid.shape[0]):
        arr.extend(r_node_haloghostnid[i, 0:r_node_haloghostnid[i, -1]])

      r_sizehaloghost = len(np.unique(arr))
      d_sizehaloghost = ld.halos.sizehaloghost

      np.testing.assert_equal(d_sizehaloghost, r_sizehaloghost)

  def test_halos_communication(self, local_domains: 'list[Domain]', reference_domain: ReferenceTables):
    if len(local_domains) <= 1:
      return
    comm = {}

    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      start = 0
      for i in range(ld.halos.neigh.shape[1]):
        neighbor = ld.halos.neigh[0, i]
        send_count = ld.halos.neigh[1, i]
        halo_int = ld.halos.halosint[start:start + send_count]
        halo_int = ld.cells.loctoglob[halo_int]
        comm[(part, neighbor)] = halo_int
        start += send_count

    # Test Communication
    for part in range(len(local_domains)):
      ld = _NumpyDomainView(local_domains[part])

      start = 0
      for i in range(ld.halos.neigh.shape[1]):
        neighbor = ld.halos.neigh[0, i]
        recv = comm[(neighbor, part)]
        recv_count = len(recv)
        halos_ext = ld.halos.halosext[start:start + recv_count]
        halos_ext = halos_ext[:, 0] # select only cells global ID.
        np.testing.assert_almost_equal(recv, halos_ext, decimal=self.decimal, err_msg=f"Send and recv differ! from {neighbor} to {part}")
        start += recv_count
