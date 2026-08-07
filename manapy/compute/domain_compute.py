import numpy as np
from manapy.backends.config import ManapyConfig
from manapy.compute._compute import _Compute

class DomainCompute:
  def __init__(self, config: ManapyConfig):
    self.config = config
    self.compute = _Compute.getComputeInstance(config)

    self.make_n_part_graph_k_way = self.compute.make_n_part_graph_k_way
    self.make_n_part_mesh_dual = self.compute.make_n_part_mesh_dual
    self.make_n_part_mesh_nodal = self.compute.make_n_part_mesh_nodal
    self.create_local_domains = self.compute.create_local_domains
    self.compute_cell_center_area_2d = self.compute.compute_cell_center_area_2d
    self.compute_cell_center_volume_3d = self.compute.compute_cell_center_volume_3d

  def create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int'):
    # Count max node cellid
    res = np.zeros(shape=nb_nodes, dtype=self.config.int_dtype)
    self.compute.count_max_node_cellid(cells, res)
    max_node_cellid = np.max(res)

    # Create node cellid
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=self.config.int_dtype)
    self.compute.create_node_cellid(cells, node_cellid)
    return node_cellid


  # LocalDomainClass.py
  def create_node_phyid(self, phy_faces: 'int[:, :]', nb_nodes: 'int'):
    # Count max node boundary faces
    # Create node boundary faceid
    return self.create_node_cellid(phy_faces, nb_nodes)

  # LocalDomainClass.py
  def create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    # Count max cell cellnid
    i_visited = np.ones(cells.shape[0], dtype=self.config.int_dtype) * -1
    max_cell_cellnid = self.compute.count_max_cell_cellnid(cells, node_cellid, i_visited)

    # Create cell cellnid
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=self.config.int_dtype)
    self.compute.create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  # Partitioning.py
  def get_max_phyid(self, nb_cells: 'int', phy_faces: 'int[:, :]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]'):
    i_visited = np.ones(shape=nb_cells, dtype=self.config.int_dtype) * -1
    cell_nb_phyid = np.zeros(shape=nb_cells, dtype=self.config.int_dtype)

    self.compute.get_cell_nb_phyid(phy_faces, node_cellid, i_visited, cell_nb_phyid)
    node_max_phyid = np.max(node_phyid[:, -1])
    cell_max_phyid = np.max(cell_nb_phyid)
    return node_max_phyid, cell_max_phyid

  # Partitioning.py
  def define_node_oldname(self, phy_faces, phy_faces_name, nb_nodes):
    node_oldname = np.zeros(shape=nb_nodes, dtype=self.config.int_dtype)
    self.compute.define_node_oldname(phy_faces, phy_faces_name, node_oldname)
    return node_oldname

  # Partitioning.py
  def create_cellfid(
    self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int'
  ):
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=self.config.int_dtype)
    # tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=self.config.int_dtype)
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)

    self.compute.create_cellfid(
      cells,
      node_cellid,
      cell_type,
      cell_cellfid
    )

    return cell_cellfid

  ############################################################################
  ############################################################################
  ############################################################################

  def create_info(self,
                   cells: 'int[:, :]',
                   node_cellid: 'int[:, :]',
                   cell_type: 'int[:]',
                   max_cell_faceid: 'int',
                   max_face_nodeid: 'int'
                   ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=self.config.int_dtype)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=self.config.int_dtype)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=self.config.int_dtype)
    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=self.config.int_dtype)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=self.config.int_dtype) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)
    faces_counter = np.zeros(shape=1, dtype=self.config.int_dtype)

    self.compute.create_info(
      cells,
      node_cellid,
      cell_type,
      tmp_cell_faces,
      tmp_size_info,
      tmp_cell_faces_map,
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid,
      faces_counter
    )

    faces = faces[:faces_counter[0]]
    face_cellid = face_cellid[:faces_counter[0]]

    return (
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid
    )

  def create_cell_info(self, cells, nodes, dim):
    nb_cells = len(cells)
    cell_volume = np.zeros(shape=nb_cells, dtype=self.config.float_dtype)
    cell_center = np.zeros(shape=(nb_cells, 3), dtype=self.config.float_dtype)
    if dim == 2:
      self.compute.compute_cell_center_area_2d(cells, nodes, cell_volume, cell_center)
    else:
      self.compute.compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center)
    return (
      cell_volume,
      cell_center
    )

  def create_face_info(self, faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]',
                        cell_center: 'float[:]', dim):
    nb_faces = len(faces)
    face_measure = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_center = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
    face_normal = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
    face_tangent = np.zeros(shape=0, dtype=self.config.float_dtype)
    face_binormal = np.zeros(shape=0, dtype=self.config.float_dtype)

    if dim == 2:
      self.compute.compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
      self.compute.compute_face_info_3d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal,
                                   face_tangent, face_binormal)
    return (
      face_measure,
      face_center,
      face_normal,
      face_tangent,
      face_binormal
    )

  def create_halo_cells(self, cells, faces, nodes, node_halos, halo_halosext, size, max_cell_halonid, max_node_haloid):
    nb_cells = len(cells)
    nb_faces = len(faces)
    nb_nodes = len(nodes)
    nb_halos = len(halo_halosext)

    if size == 1:
      # give size to cell_halonid, face_haloid and node_haloid to keep the multiprocessing code as it is
      cell_halonid = np.zeros(shape=(nb_cells, 1), dtype=self.config.int_dtype)
      face_haloid = np.ones(shape=nb_faces, dtype=self.config.int_dtype) * -1
      node_haloid = np.zeros(shape=(nb_nodes, 1), dtype=self.config.int_dtype)
    else:
      cell_halonid = np.zeros(shape=(nb_cells, max_cell_halonid + 1), dtype=self.config.int_dtype)
      face_haloid = np.zeros(shape=nb_faces, dtype=self.config.int_dtype)
      node_haloid = np.zeros(shape=(nb_nodes, max_node_haloid + 1), dtype=self.config.int_dtype)
      b_visited = np.zeros(shape=nb_halos, dtype=np.int8)

      self.compute.create_halo_cells(cells, faces, node_halos, node_haloid, b_visited, cell_halonid, face_haloid)

    return (
      cell_halonid,
      face_haloid,
      node_haloid
    )

  def define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 face_haloid: 'int[:]',
                                 node_haloid: 'int[:, :]',
                                 node_oldname: 'int[:]',
                                 nb_nodes
                                 ):
    face_name = np.zeros(shape=faces.shape[0], dtype=self.config.int_dtype)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=self.config.int_dtype)
    phyid_to_faceid = np.ones(shape=phy_faces.shape[0], dtype=self.config.int_dtype) * -1
    face_to_phyid = np.ones(shape=faces.shape[0], dtype=self.config.int_dtype) * -1

    node_name = node_oldname.copy()
    if node_haloid.shape[0] != 0:
      node_name[node_haloid[:, -1] != 0] = 10

    node_phyid = self.create_node_phyid(phy_faces, nb_nodes)

    self.compute.define_face_name(phy_faces, phy_faces_name, faces, node_phyid, face_haloid, face_oldname, face_name, phyid_to_faceid, face_to_phyid)

    return (
      face_oldname,
      face_name,
      node_name,
      phyid_to_faceid,
      face_to_phyid
    )

  def create_ghost_info(self, cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]',
                         face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]',
                         face_measure: 'float[:]', faces: 'int[:, :]', nodes: 'float[:, :]', phy_faces: 'int[:, :]',
                         node_cellid: 'int[:, :]', phyid_to_faceid: 'int[:]', nb_phy_faces, phy_faces_name, dim):

    ghost_info_size = nb_phy_faces

    # ---- bf_cellid
    bf_cellid = np.zeros(shape=(ghost_info_size, 2), dtype=self.config.int_dtype)
    intersect = np.zeros(shape=2, dtype=self.config.int_dtype)
    self.compute.create_bf_cellid(phy_faces, node_cellid, phyid_to_faceid, cell_faceid, intersect, bf_cellid)

    # Periodic faces (names 11/22/33/44/55/66) are NOT physical boundaries: their
    # partner cell is delivered through the (periodic) halo, so they must not get a
    # ghost. An unvalued periodic ghost would bias the node interpolation of no-BC
    # variables toward 0 (VTK). Mark them invalid so both ghost kernels skip them.
    pn = phy_faces_name
    per = (pn == 11) | (pn == 22) | (pn == 33) | (pn == 44) | (pn == 55) | (pn == 66)
    if np.any(per):
      bf_cellid[per] = -1

    # ---- ghost_info_flt, ghost_info_int
    ghost_info_data_size_flt = 10  # (ghostcenter_x&y&z, gamma, face_center_x&y&z, face_normal_x&y&z)
    ghost_info_data_size_int = 5  # (cell_id, face index inside the cell, face_oldname, cell global id, face_id)
    ghost_info_flt = np.zeros(shape=(ghost_info_size, ghost_info_data_size_flt), dtype=self.config.float_dtype)
    ghost_info_int = np.zeros(shape=(ghost_info_size, ghost_info_data_size_int), dtype=self.config.int_dtype)

    self.compute.create_ghost_info(bf_cellid, cell_center, cell_faceid, cell_loctoglob, faces, nodes, face_oldname,
                              face_normal, face_center, face_measure, ghost_info_int, ghost_info_flt, dim)

    return ghost_info_int, ghost_info_flt

  def create_ghost_tables(self, ghost_info_int: 'int[:, :]', node_cellid: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', max_node_phyid, max_cell_phyid):

    max_cell_ghostnid = max_cell_phyid
    nb_cells = len(cell_faceid)
    nb_nodes = len(node_cellid)
    nb_faces = len(faces)

    cell_ghostnid = np.zeros(shape=(nb_cells, max_cell_ghostnid + 1), dtype=self.config.int_dtype)
    node_ghostid = np.zeros(shape=(nb_nodes, max_node_phyid + 1), dtype=self.config.int_dtype)

    ghost_i_visited = np.ones(shape=nb_faces, dtype=self.config.int_dtype) * -1
    self.compute.create_ghost_tables(ghost_info_int, faces, cell_faceid, node_cellid, ghost_i_visited, node_ghostid, cell_ghostnid)

    return (
      node_ghostid,
      cell_ghostnid
    )

  def create_halo_ghost_tables(self, ext_ghost_info_int: 'float[:, :]', node_halophyid: 'int[:]', cell_halophyid: 'int[:]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', max_cell_halophyid, max_node_halophyid, size, nb_nodes, nb_cells):
    # node_halophyid / cell_halophyid are FLAT run-length lists
    # ([id, size, phyid...] repeated), not per-node/per-cell tables -- their
    # length says nothing about the mesh size (it is 0 in serial). The output
    # tables are indexed by local node/cell id, so they must be sized by the
    # mesh counts.
    if size == 1:
      # give size to cell_haloghostnid and node_haloghostid to keep the multiprocessing code as it is
      cell_haloghostid = np.zeros(shape=(nb_cells, 1), dtype=self.config.int_dtype)
      node_haloghostid = np.zeros(shape=(nb_nodes, 1), dtype=self.config.int_dtype)
    else:
      cell_haloghostid = np.zeros(shape=(nb_cells, max_cell_halophyid + 1), dtype=self.config.int_dtype)
      node_haloghostid = np.zeros(shape=(nb_nodes, max_node_halophyid + 1), dtype=self.config.int_dtype)
      # It will also update ext_ghost_info_int[0] from cell_id to haloext of the cell
      self.compute.create_halo_ghost_tables(ext_ghost_info_int, node_halophyid, cell_halophyid, node_haloid, halo_halosext, cell_haloghostid, node_haloghostid)

    return (
      cell_haloghostid,
      node_haloghostid
    )

  def face_gradient_info(self, face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, cell_shift, dim):
    nb_faces = len(faces)

    face_air_diamond = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param1 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param2 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param3 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param4 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_f1 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f2 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f3 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f4 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)

    if dim == 2:
      self.compute.face_gradient_info_2d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_param4, face_f1, face_f2, face_f3, face_f4, cell_shift)
    else:
      self.compute.face_gradient_info_3d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_f1, face_f2, cell_shift)

    return (
      face_air_diamond,
      face_param1,
      face_param2,
      face_param3,
      face_param4,
      face_f1,
      face_f2,
      face_f3,
      face_f4
    )

  def fv_face_geometry(self, face_cellid, face_name, face_normal, face_center, face_haloid, cell_center, halo_centvol, cell_shift):
    nb_faces = len(face_normal)

    face_fv_coeff = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corrx = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corry = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corrz = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_weight_left = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    self.compute.fv_face_geometry(face_cellid, face_name, face_normal, face_center, face_haloid, cell_center, halo_centvol, cell_shift, face_fv_coeff, face_fv_corrx, face_fv_corry, face_fv_corrz, face_fv_weight_left)

    return (
      face_fv_coeff,
      face_fv_corrx,
      face_fv_corry,
      face_fv_corrz,
      face_fv_weight_left
    )

  def variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, cell_shift, dim):
    nb_nodes = len(nodes)

    node_R_x = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_R_y = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_R_z = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_x = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_y = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_z = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_number = np.zeros(nb_nodes, dtype=self.config.int_dtype)

    if dim == 2:
      self.compute.variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      self.compute.variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

    return (
      node_R_x,
      node_R_y,
      node_R_z,
      node_lambda_x,
      node_lambda_y,
      node_lambda_z,
      node_number
    )

  def create_normal_face_of_cell(self, cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int[:,:]', face_normal: 'float[:,:]', max_cell_faceid):
    nb_cells = len(cell_center)

    cell_nf = np.zeros(shape=(nb_cells, max_cell_faceid, 3), dtype=self.config.float_dtype)
    self.compute.create_normal_face_of_cell(cell_center, face_center, cell_faceid, face_normal, cell_nf)
    return cell_nf

  def dist_ortho_function_2d(self, d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]', dim):
    nb_faces = len(face_normal)
    face_dist_ortho = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    if dim == 2:
      self.compute.dist_ortho_function_2d(d_innerfaces, d_boundaryfaces, face_cellid, cell_center, face_center, face_normal, face_dist_ortho)
    return face_dist_ortho

  def build_periodic_samerank(self, nodes, node_cellid, faces, face_name, face_center, face_cellid, cell_shift, dim):
    # SAME-RANK periodic pairing. For periodic pairs whose BOTH sides live in this
    # subdomain it wires:
    #   * face_cellid[face][1] = partner cell + cell_shift (image across boundary),
    #     so the flux and geometry kernels treat the periodic face like interior;
    #   * node_periodicid[node] = partner node's cells, so the VTK node
    #     interpolation sees both sides (else its least-squares stencil is
    #     one-sided -> singular).
    # Sign of the shift by the boundary a cell touches: x=0/y=0/z=0 (tags
    # 11/44/55) -> +L on that component, x=Lx/y=Ly/z=Lz (tags 22/33/66) -> -L.
    #
    # CROSS-RANK pairs are NOT seen here: the C++ partitioner
    # (handle_periodic_faces) already turned them into halo faces (face_name==10,
    # translated halo_centvol), so they no longer carry tags 11/22/33/44/55/66.
    #
    # The matching itself is done in COMPILED kernels (domain_compute:
    # pair_periodic_faces / match_periodic_nodes / fill_node_periodicid). They are
    # sort-based (no Python dict) and O(#periodic boundary faces/nodes), so this
    # stays cheap even in 3D where those are O(N^2) surface quantities.
    #
    # Lx/Ly/Lz come from the LOCAL node extent, which is EXACT for same-rank
    # pairs: a same-rank pair needs both boundaries of that axis in THIS
    # subdomain, so it spans the full box on that axis (max-min == global length).
    nb_nodes = len(nodes)
    node_periodicid = np.zeros((nb_nodes, 2), dtype=self.config.int_dtype)
    fname = face_name
    if not np.any((fname == 11) | (fname == 22) | (fname == 33) |
                  (fname == 44) | (fname == 55) | (fname == 66)):
      return node_periodicid

    nmin = np.array([nodes[:, 0].min(), nodes[:, 1].min(),
                     nodes[:, 2].min()], dtype=self.config.float_dtype)
    nmax = np.array([nodes[:, 0].max(), nodes[:, 1].max(),
                     nodes[:, 2].max()], dtype=self.config.float_dtype)
    dtol = 1e-6 * float(np.max(nmax - nmin))   # transverse-match tolerance
    Lx = float(nmax[0] - nmin[0])
    Ly = float(nmax[1] - nmin[1])
    Lz = float(nmax[2] - nmin[2])
    t2 = 2 if dim == 3 else -1

    # faces: (name_lo, name_hi, taxis0, taxis1, shift_axis, L)
    fdirs = [(11, 22, 1, t2, 0, Lx), (44, 33, 0, t2, 1, Ly)]
    if dim == 3:
      fdirs.append((55, 66, 0, 1, 2, Lz))
    for (name_lo, name_hi, t0, t1, sax, L) in fdirs:
      r = self.compute.pair_periodic_faces(face_name, face_center,
                                      face_cellid, cell_shift, nmin,
                                      name_lo, name_hi, t0, t1, sax, L, dtol)
      if r < 0:
        raise ValueError(
          "same-rank periodic face pairing failed (code %d) for tags %d/%d: a "
          "periodic face has no local partner (cross-rank should already be a "
          "halo) or the two boundaries are non-conforming." % (r, name_lo, name_hi))

    # nodes: per-node periodic-boundary bitmask (from the periodic faces), then
    # accumulate partner cells per periodic axis into node_periodicid. An edge or
    # corner node carries several bits, so it collects partners from EVERY
    # direction it touches -- fixing the 3D one-sided-stencil div-by-zero that a
    # single node_oldname tag caused. Width 3*node_cellid gives room for up to 3
    # directions of partners.
    node_bits = np.zeros(nb_nodes, dtype=self.config.int_dtype)
    self.compute.node_periodic_bits(faces, face_name, node_bits)
    npid = np.zeros((nb_nodes, 3 * node_cellid.shape[1]),
                    dtype=self.config.int_dtype)
    node_fill = np.zeros(nb_nodes, dtype=self.config.int_dtype)
    # bits: 1=x-lo(11) 2=x-hi(22) 4=y-hi(33) 8=y-lo(44) 16=z-lo(55) 32=z-hi(66)
    self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                               node_fill, nmin, 1, 2, 1, t2, dtol)   # x -> (y[,z])
    self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                               node_fill, nmin, 8, 4, 0, t2, dtol)   # y -> (x[,z])
    if dim == 3:
      self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                                 node_fill, nmin, 16, 32, 0, 1, dtol)  # z -> (x,y)
    if np.any(node_fill > 0):
      return npid
    return node_periodicid







