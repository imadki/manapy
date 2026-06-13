import numpy as np
from mpi4py import MPI
from manapy.backends.debug import log_step
import manapy.domain.domain_compute as compute
import manapy.domain.utils as utils
from manapy.domain.LocalDomainInterface import LocalDomainInterface
import sys
from manapy.comms.NeighborCommunication import NeighborCommunication
import manapy.backends.types as types

class LocalDomain:

  def __init__(self, local_domain_struct: 'LocalDomainInterface', rank: 'int', size: 'int', backend=None):
    if local_domain_struct is None:
      return

    from manapy.backends import get_backend
    self.backend = get_backend("cpu") if backend is None else (get_backend(backend) if isinstance(backend, str) else backend)
    self.rank = rank
    self.size = size
    self.dim = local_domain_struct.dim
    # Compile the dimension-specific domain kernels once (only the used dim).
    # Called uniformly on all ranks here to stay barrier-safe.
    compute.setup(self.dim)
    self.nodes = local_domain_struct.nodes
    self.cells = local_domain_struct.cells
    self.cells_type = local_domain_struct.cells_type
    self.phy_faces = local_domain_struct.phy_faces
    self.phy_faces_name = local_domain_struct.phy_faces_name
    self.cell_loctoglob = local_domain_struct.cell_loctoglob
    self.cell_tc = local_domain_struct.cell_tc
    self.node_loctoglob = local_domain_struct.node_loctoglob
    self.node_oldname = local_domain_struct.node_oldname
    self.halo_neighsub = local_domain_struct.halo_neighsub
    self.halo_halosint = local_domain_struct.halo_halosint
    self.halo_halosext = local_domain_struct.halo_halosext
    self.halo_centvol = local_domain_struct.halo_centvol
    self.node_halos = local_domain_struct.node_halos
    self.phyid_neighbor = local_domain_struct.phyid_neighbor
    self.phyid_recv = local_domain_struct.phyid_recv
    self.phyid_send = local_domain_struct.phyid_send
    self.node_halophyid = local_domain_struct.node_halophyid
    self.cell_halophyid = local_domain_struct.cell_halophyid
    self.max_cell_nodeid = local_domain_struct.max_cell_nodeid
    self.max_cell_faceid = local_domain_struct.max_cell_faceid
    self.max_face_nodeid = local_domain_struct.max_face_nodeid
    self.max_node_haloid = local_domain_struct.max_node_haloid
    self.max_cell_halonid = local_domain_struct.max_cell_halonid
    self.max_node_phyid = local_domain_struct.max_node_phyid
    self.max_node_halophyid = local_domain_struct.max_node_halophyid
    self.max_cell_phyid = local_domain_struct.max_cell_phyid
    self.max_cell_halophyid = local_domain_struct.max_cell_halophyid
    self.nb_nodes = types.np_int_type(len(self.nodes))
    self.nb_cells = types.np_int_type(len(self.cells))
    self.nb_phy_faces = types.np_int_type(len(self.phy_faces))
    self.test = False # debug attribute


    log_step.log("Prepare communication")
    tmp = self.prepare_comm(self.halo_neighsub, self.halo_halosint)
    self.halo_comm : NeighborCommunication = tmp[0]
    self.phy_faces_comm : NeighborCommunication = tmp[1]
    log_step.out()

    log_step.log("bounds")
    self.bounds = self._define_bounds(self.nodes)
    log_step.out()

    log_step.log("node_cellid")
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes)
    log_step.out()

    log_step.log("cell_cellnid")
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid)
    log_step.out()

    log_step.log("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid
    ) = self._create_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    self.nb_faces = len(self.faces)
    log_step.out()


    log_step.log("Create cell volume and center")
    (
      self.cell_volume,
      self.cell_center
    ) = self._create_cell_info(self.cells, self.nodes)
    log_step.out()

    log_step.log("Face measure face center face normal")
    (
      self.face_measure,
      self.face_center,
      self.face_normal,
      self.face_tangent, # only in 3D, shape is 0 in 2D
      self.face_binormal # only in 3D, shape is 0 in 2D
    ) = self._create_face_info(self.faces, self.nodes, self.face_cellid, self.cell_center)
    log_step.out()

    log_step.log("cell_halonid, face_haloid, node_haloid")
    (
      self.cell_halonid,
      self.face_haloid,
      self.node_haloid
    ) = self._create_halo_cells(self.cells, self.faces, self.nodes, self.node_halos, self.halo_halosext)
    log_step.out()

    log_step.log("Create face and node names")
    (
      self.face_oldname,
      self.face_name,
      self.node_name,
      self.phyid_to_faceid,
      self.face_to_phyid,
    ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid, self.node_haloid, self.node_oldname)
    log_step.out()


    log_step.log("_create_shared_ghost_info")
    (self.ghost_info_int, self.ghost_info_flt) = self._create_ghost_info(self.cell_center, self.cell_faceid, self.cell_loctoglob, self.face_oldname, self.face_normal, self.face_center, self.face_measure, self.faces, self.nodes, self.phy_faces, self.node_cellid, self.phyid_to_faceid)
    log_step.out()

    log_step.log("_share_ghost_info_flt and _share_ghost_info_int")
    (
      self.ext_ghost_info_flt,
      self.ext_ghost_info_int
    ) = self.phy_faces_exchange(self.phy_faces_comm, self.ghost_info_int, self.ghost_info_flt)
    log_step.out()

    log_step.log("_create_ghost_tables")
    (
      self.node_ghostid,
      self.cell_ghostid
    ) = self._create_ghost_tables(self.ghost_info_int, self.node_cellid, self.faces, self.cell_faceid)
    log_step.out()

    log_step.log("_create_halo_ghost_tables")
    (
      self.cell_haloghostid,
      self.node_haloghostid
    ) = self._create_halo_ghost_tables(self.ext_ghost_info_int, self.node_halophyid, self.cell_halophyid, self.node_haloid, self.halo_halosext)
    log_step.out()

    # Check if mesh is will constructed
    log_step.log("_check_phy_faces")
    self._check_phy_faces(self.face_cellid, self.face_haloid)
    log_step.out()

    ## TODO the use of this tables !?
    self.node_periodicid = np.zeros((self.nb_nodes, 2), dtype=types.np_int_type)
    self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=types.np_int_type)
    self.cell_periodicfid = np.zeros(self.nb_cells, dtype=types.np_int_type)
    self.cell_shift = np.zeros((self.nb_cells, 3), dtype=types.np_float_type)

    log_step.log("_face_gradient_info")
    (
      self.face_air_diamond,
      self.face_param1,
      self.face_param2,
      self.face_param3,
      self.face_param4,
      self.face_f1,
      self.face_f2,
      self.face_f3,
      self.face_f4 # only on 2D
    ) = self._face_gradient_info(self.face_cellid, self.faces, self.face_to_phyid, self.ghost_info_flt, self.face_name, self.face_normal, self.cell_center, self.halo_centvol, self.face_haloid, self.nodes, self.cell_shift)
    log_step.out()

    log_step.log("_variables")
    (
      self.node_R_x,
      self.node_R_y,
      self.node_R_z, # Only on 3D
      self.node_lambda_x,
      self.node_lambda_y,
      self.node_lambda_z, # Only on 3D
      self.node_number,
    ) = self._variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid, self.node_haloghostid, self.node_periodicid, self.nodes, self.node_oldname, self.ghost_info_flt, self.ext_ghost_info_flt, self.halo_centvol, self.cell_shift)
    log_step.out()

    log_step.log("_update_boundaries")
    (
      self.innerfaces,
      self.infaces,
      self.outfaces,
      self.upperfaces,
      self.bottomfaces,
      self.halofaces,
      self.periodicinfaces,
      self.periodicoutfaces,
      self.periodicupperfaces,
      self.periodicbottomfaces,
      self.boundaryfaces,
      self.periodicboundaryfaces,
      self.innernodes,
      self.innodes,
      self.outnodes,
      self.uppernodes,
      self.bottomnodes,
      self.halonodes,
      self.periodicinnodes,
      self.periodicoutnodes,
      self.periodicuppernodes,
      self.periodicbottomnodes,
      self.boundarynodes,
      self.periodicboundarynodes,
      self.frontfaces, # only in 3D, shape is 0 in 2D
      self.backfaces, # only in 3D, shape is 0 in 2D
      self.periodicfrontfaces, # only in 3D, shape is 0 in 2D
      self.periodicbackfaces, # only in 3D, shape is 0 in 2D
      self.frontnodes, # only in 3D, shape is 0 in 2D
      self.backnodes, # only in 3D, shape is 0 in 2D
      self.periodicfrontnodes, # only in 3D, shape is 0 in 2D
      self.periodicbacknodes, # only in 3D, shape is 0 in 2D
    ) = self._update_boundaries(self.face_name, self.node_name)
    log_step.out()

    log_step.log("_define_BCs")
    self.BCs = self._define_BCs(self.periodicinfaces, self.periodicupperfaces, self.periodicfrontfaces)
    log_step.out()

    log_step.log("_create_normal_face_of_cell_2d")
    self.cell_nf = self._create_normal_face_of_cell(self.cell_center, self.face_center, self.cell_faceid, self.face_normal)
    log_step.out()

    log_step.log("_dist_ortho_function_2d")
    # only in 2D, shape is 0 in 3D
    self.face_dist_ortho = self._dist_ortho_function_2d(self.innerfaces, self.boundaryfaces, self.face_cellid, self.cell_center, self.face_center, self.face_normal)
    log_step.out()

  def prepare_comm(self, halo_neighsub, halo_halosint):
    if self.size == 1:
      return NeighborCommunication(None, None, None), NeighborCommunication(None, None, None)

    halo_cells_comm = NeighborCommunication(
      neighbors=halo_neighsub[0],
      send_counts=halo_neighsub[1],
      send_indices=halo_halosint
    )

    phy_faces_comm = NeighborCommunication(
      neighbors=self.phyid_neighbor[:, 0],
      send_counts=self.phyid_neighbor[:, 1],
      send_indices=self.phyid_send
    )

    return halo_cells_comm, phy_faces_comm

  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int'):
    return utils.create_node_cellid(cells, nb_nodes)


  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    return utils.create_cell_cellnid(cells, node_cellid)


  def _create_info(self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int'
  ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=types.np_int_type)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=types.np_int_type)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=types.np_int_type)
    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=types.np_int_type)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=types.np_int_type)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=types.np_int_type) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=types.np_int_type)
    faces_counter = np.zeros(shape=1, dtype=types.np_int_type)

    compute.create_info(
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


  def _create_cell_info(self, cells, nodes):
    nb_cells = len(cells)
    cell_volume = np.zeros(shape=nb_cells, dtype=types.np_float_type)
    cell_center = np.zeros(shape=(nb_cells, 3), dtype=types.np_float_type)
    if self.dim == 2:
      compute.compute_cell_center_volume_2d(cells, nodes, cell_volume, cell_center)
    else:
      compute.compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center)
    return (
      cell_volume,
      cell_center
    )

  def _create_face_info(self, faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]'):
    nb_faces = len(faces)
    face_measure = np.zeros(shape=nb_faces, dtype=types.np_float_type)
    face_center = np.zeros(shape=(nb_faces, 3), dtype=types.np_float_type)
    face_normal = np.zeros(shape=(nb_faces, 3), dtype=types.np_float_type)
    face_tangent = np.zeros(shape=0, dtype=types.np_float_type)
    face_binormal = np.zeros(shape=0, dtype=types.np_float_type)

    if self.dim == 2:
      compute.compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=types.np_float_type)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=types.np_float_type)
      compute.compute_face_info_3d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal, face_tangent, face_binormal)
    return (
      face_measure,
      face_center,
      face_normal,
      face_tangent,
      face_binormal
    )

  def _create_halo_cells(self, cells, faces, nodes, node_halos, halo_halosext):
    nb_cells = len(cells)
    nb_faces = len(faces)
    nb_nodes = len(nodes)
    nb_halos = len(halo_halosext)

    if self.size == 1:
      # give size to cell_halonid, face_haloid and node_haloid to keep the multiprocessing code as it is
      cell_halonid = np.zeros(shape=(nb_cells, 1), dtype=types.np_int_type)
      face_haloid = np.ones(shape=nb_faces, dtype=types.np_int_type) * -1
      node_haloid = np.zeros(shape=(nb_nodes, 1), dtype=types.np_int_type)
    else:
      cell_halonid = np.zeros(shape=(nb_cells, self.max_cell_halonid + 1), dtype=types.np_int_type)
      face_haloid = np.zeros(shape=nb_faces, dtype=types.np_int_type)
      node_haloid = np.zeros(shape=(nb_nodes, self.max_node_haloid + 1), dtype=types.np_int_type)
      b_visited = np.zeros(shape=nb_halos, dtype=np.int8)

      compute.create_halo_cells(cells, faces, node_halos, node_haloid, b_visited, cell_halonid, face_haloid)

    return (
      cell_halonid,
      face_haloid,
      node_haloid
    )



  def _define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 face_haloid: 'int[:]',
                                 node_haloid: 'int[:, :]',
                                 node_oldname: 'int[:]'
                                 ):
    nb_nodes = self.nb_nodes
    face_name = np.zeros(shape=faces.shape[0], dtype=types.np_int_type)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=types.np_int_type)
    phyid_to_faceid = np.ones(shape=phy_faces.shape[0], dtype=types.np_int_type) * -1
    face_to_phyid = np.ones(shape=faces.shape[0], dtype=types.np_int_type) * -1

    node_name = node_oldname.copy()
    if node_haloid.shape[0] != 0:
      node_name[node_haloid[:, -1] != 0] = 10

    node_phyid = utils.create_node_phyid(phy_faces, nb_nodes)

    compute.define_face_name(phy_faces, phy_faces_name, faces, node_phyid, face_haloid, face_oldname, face_name, phyid_to_faceid, face_to_phyid)

    return (
      face_oldname,
      face_name,
      node_name,
      phyid_to_faceid,
      face_to_phyid
    )


  def _create_ghost_info(self, cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', faces: 'int[:, :]', nodes: 'float[:, :]', phy_faces: 'int[:, :]', node_cellid: 'int[:, :]', phyid_to_faceid: 'int[:]'):

    ghost_info_size = self.nb_phy_faces

    # ---- bf_cellid
    bf_cellid = np.zeros(shape=(ghost_info_size, 2), dtype=types.np_int_type)
    intersect = np.zeros(shape=2, dtype=types.np_int_type)
    compute.create_bf_cellid(phy_faces, node_cellid, phyid_to_faceid, cell_faceid, intersect, bf_cellid)

    # ---- ghost_info_flt, ghost_info_int
    ghost_info_data_size_flt = 10 # (ghostcenter_x&y&z, gamma, face_center_x&y&z, face_normal_x&y&z)
    ghost_info_data_size_int = 5 # (cell_id, face index inside the cell, face_oldname, cell global id, face_id)
    ghost_info_flt = np.zeros(shape=(ghost_info_size, ghost_info_data_size_flt), dtype=types.np_float_type)
    ghost_info_int = np.zeros(shape=(ghost_info_size, ghost_info_data_size_int), dtype=types.np_int_type)


    compute.create_ghost_info(bf_cellid, cell_center, cell_faceid, cell_loctoglob, faces, nodes, face_oldname, face_normal, face_center, face_measure, ghost_info_int, ghost_info_flt, self.dim)

    return ghost_info_int, ghost_info_flt

  def _create_ghost_tables(self, ghost_info_int: 'int[:, :]', node_cellid: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]'):

    max_node_phyid = self.max_node_phyid
    max_cell_ghostnid = self.max_cell_phyid

    cell_ghostnid = np.zeros(shape=(self.nb_cells, max_cell_ghostnid + 1), dtype=types.np_int_type)
    node_ghostid = np.zeros(shape=(self.nb_nodes, max_node_phyid + 1), dtype=types.np_int_type)

    ghost_i_visited = np.ones(shape=self.nb_faces, dtype=types.np_int_type) * -1
    compute.create_ghost_tables(ghost_info_int, faces, cell_faceid, node_cellid, ghost_i_visited, node_ghostid, cell_ghostnid)

    return (
      node_ghostid,
      cell_ghostnid
    )

  def phy_faces_exchange(self, phy_faces_comm, ghost_info_int: 'int[:, :]', ghost_info_flt: 'float[:, :]'):
    ext_ghost_info_flt = np.zeros(shape=(0, 0), dtype=types.np_float_type)
    ext_ghost_info_int = np.zeros(shape=(0, 0), dtype=types.np_int_type)
    if self.size != 1:
      # Return ghost_ext_info_int # (0=haloext, 1=face_oldname, 2=cell global id)
      ghost_info_int = ghost_info_int[:, [0, 2, 3]] # only (local_cellid, face_oldname, cell_global_id)
      # local_cellid it will be replaced with haloext after exchange later in _create_halo_ghost_tables
      ext_ghost_info_flt = phy_faces_comm.exchange(ghost_info_flt)
      ext_ghost_info_int = phy_faces_comm.exchange(ghost_info_int)
    return (
      ext_ghost_info_flt,
      ext_ghost_info_int
    )

  @staticmethod
  def _local_share_ghost_info(local_domains: 'LocalDomain[:]'):
    if len(local_domains) == 1:
      local_domains[0].ext_ghost_info_flt = np.zeros(shape=(0, 0), dtype=types.np_float_type)
      local_domains[0].ext_ghost_info_int = np.zeros(shape=(0, 0), dtype=types.np_int_type)
      return

    size = len(local_domains)
    recv_data = np.ndarray(shape=(size, size), dtype=object)
    recv_data.fill((None, None))

    # ------------------------------------------------------------------
    # 1. Send
    # ------------------------------------------------------------------
    for rank in range(size):
      domain : LocalDomain = local_domains[rank]
      ghost_info_flt = domain.ghost_info_flt
      ghost_info_int = domain.ghost_info_int

      send_displs = np.insert(np.cumsum(domain.phyid_neighbor[:, 1]), 0, 0)
      neighbors = domain.phyid_neighbor[:, 0]
      phyid_send = domain.phyid_send
      for i in range(neighbors.shape[0]):
        dest_part = neighbors[i]
        a = send_displs[i]
        b = send_displs[i + 1]
        data_indices = phyid_send[a:b]
        data_flt = ghost_info_flt[data_indices]
        data_int = ghost_info_int[data_indices]
        data_int = data_int[:, [0, 2, 3]]
        recv_data[rank][dest_part] = (data_indices, data_int, data_flt)

    # ------------------------------------------------------------------
    # 2. Receive
    # ------------------------------------------------------------------
    for rank in range(size):
      domain : LocalDomain = local_domains[rank]
      ext_ghost_info_flt = []
      ext_ghost_info_int = []

      neighbors = domain.phyid_neighbor[:, 0]
      for i in range(neighbors.shape[0]):
        sender = neighbors[i]
        (data_indices, data_int, data_flt) = recv_data[sender][rank]
        ext_ghost_info_flt.extend(data_flt)
        ext_ghost_info_int.extend(data_int)

      if len(ext_ghost_info_flt) == 0:
        ext_ghost_info_flt = [[]]
      if len(ext_ghost_info_int) == 0:
        ext_ghost_info_int = [[]]
      domain.ext_ghost_info_flt = np.array(ext_ghost_info_flt, dtype=types.np_float_type)
      domain.ext_ghost_info_int = np.array(ext_ghost_info_int, dtype=types.np_int_type)


  def _create_halo_ghost_tables(self, ext_ghost_info_int: 'float[:, :]', node_halophyid: 'int[:, :]', cell_halophyid: 'int[:]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]'):
    nb_nodes = self.nb_nodes
    nb_cells = self.nb_cells
    max_cell_halophyid = self.max_cell_halophyid
    max_node_halophyid = self.max_node_halophyid

    if self.size == 1:
      # give size to cell_haloghostnid and node_haloghostid to keep the multiprocessing code as it is
      cell_haloghostid = np.zeros(shape=(nb_cells, 1), dtype=types.np_int_type)
      node_haloghostid = np.zeros(shape=(nb_nodes, 1), dtype=types.np_int_type)
    else:
      cell_haloghostid = np.zeros(shape=(nb_cells, max_cell_halophyid + 1), dtype=types.np_int_type)
      node_haloghostid = np.zeros(shape=(nb_nodes, max_node_halophyid + 1), dtype=types.np_int_type)
      # It will also update ext_ghost_info_int[0] from cell_id to haloext of the cell
      compute.create_halo_ghost_tables(ext_ghost_info_int, node_halophyid, cell_halophyid, node_haloid, halo_halosext, cell_haloghostid, node_haloghostid)

    return (
      cell_haloghostid,
      node_haloghostid
    )

  def _check_phy_faces(self, face_cellid, face_haloid):
    if self.size == 1:
      ext_faces = np.sum(face_cellid[:, 1] == -1)
    else:
      ext_faces = np.sum((face_cellid[:, 1] == -1) & (face_haloid == -1))
    total_ext_faces = MPI.COMM_WORLD.reduce(ext_faces, op=MPI.SUM, root=0)
    if total_ext_faces == 0:
      print("No physical faces found", file=sys.stderr)
      MPI.COMM_WORLD.Abort(1)
    total_phy_faces = MPI.COMM_WORLD.reduce(self.nb_phy_faces, op=MPI.SUM, root=0)
    if self.rank == 0 and total_ext_faces != total_phy_faces:
      print(f"Mess-constructed mesh number of physical faces are not equal to boundary faces. nb_physical={total_phy_faces} nb_boundary={total_ext_faces}", file=sys.stderr)
      MPI.COMM_WORLD.Abort(1)
    MPI.COMM_WORLD.barrier()

  @staticmethod
  def _local_check_phy_faces(ld: 'LocalDomain[:]'):
    total_ext_faces = 0
    total_phy_faces = 0
    for i in range(len(ld)):
      if ld[i].size == 1:
        ext_faces = np.sum(ld[i].face_cellid[:, 1] == -1)
      else:
        ext_faces = np.sum((ld[i].face_cellid[:, 1] == -1) & (ld[i].face_haloid == -1))
      total_ext_faces += ext_faces
      total_phy_faces += len(ld[i].phy_faces)
    if total_ext_faces == 0:
      raise RuntimeError("No physical faces found")
    if total_ext_faces != total_phy_faces:
      raise RuntimeError(f"Mess-constructed mesh number of physical faces are not equal to boundary faces. nb_physical={total_phy_faces} nb_boundary={total_ext_faces}")


  def _face_gradient_info(self, face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, cell_shift):

    face_air_diamond = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    face_param1 = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    face_param2 = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    face_param3 = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    face_param4 = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    face_f1 = np.zeros(shape=(self.nb_faces, self.dim), dtype=types.np_float_type)
    face_f2 = np.zeros(shape=(self.nb_faces, self.dim), dtype=types.np_float_type)
    face_f3 = np.zeros(shape=(self.nb_faces, self.dim), dtype=types.np_float_type)
    face_f4 = np.zeros(shape=(self.nb_faces, self.dim), dtype=types.np_float_type)

    if self.dim == 2:
      compute.face_gradient_info_2d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_param4, face_f1, face_f2, face_f3, face_f4, cell_shift)
    else:
      compute.face_gradient_info_3d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_f1, face_f2, cell_shift)

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

  def _variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, cell_shift):

    node_R_x = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_R_y = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_R_z = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_lambda_x = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_lambda_y = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_lambda_z = np.zeros(self.nb_nodes, dtype=types.np_float_type)
    node_number = np.zeros(self.nb_nodes, dtype=types.np_int_type)

    if self.dim == 2:
      compute.variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      compute.variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

    return (
      node_R_x,
      node_R_y,
      node_R_z,
      node_lambda_x,
      node_lambda_y,
      node_lambda_z,
      node_number
    )

  def _create_normal_face_of_cell(self, cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int[:,:]', face_normal: 'float[:,:]'):

    cell_nf = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 3), dtype=types.np_float_type)
    compute.create_normal_face_of_cell(cell_center, face_center, cell_faceid, face_normal, cell_nf)
    return cell_nf

  def _dist_ortho_function_2d(self, d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]'):

    face_dist_ortho = np.zeros(shape=self.nb_faces, dtype=types.np_float_type)
    if self.dim == 2:
      compute.dist_ortho_function_2d(d_innerfaces, d_boundaryfaces, face_cellid, cell_center, face_center, face_normal, face_dist_ortho)
    return face_dist_ortho

  def _update_boundaries(self, face_name, node_name):

    innerfaces = np.where(face_name == 0)[0].astype(types.np_int_type)
    infaces = np.where(face_name == 1)[0].astype(types.np_int_type)
    outfaces = np.where(face_name == 2)[0].astype(types.np_int_type)
    upperfaces = np.where(face_name == 3)[0].astype(types.np_int_type)
    bottomfaces = np.where(face_name == 4)[0].astype(types.np_int_type)
    halofaces = np.where(face_name == 10)[0].astype(types.np_int_type)
    if self.size == 1:
      halofaces = np.asarray([], dtype=types.np_int_type)

    periodicinfaces = np.where(face_name == 11)[0].astype(types.np_int_type)
    periodicoutfaces = np.where(face_name == 22)[0].astype(types.np_int_type)
    periodicupperfaces = np.where(face_name == 33)[0].astype(types.np_int_type)
    periodicbottomfaces = np.where(face_name == 44)[0].astype(types.np_int_type)

    boundaryfaces = np.concatenate([infaces, outfaces, bottomfaces, upperfaces])
    periodicboundaryfaces = np.concatenate([periodicinfaces, periodicoutfaces, periodicbottomfaces, periodicupperfaces])

    innernodes = np.where(node_name == 0)[0].astype(types.np_int_type)
    innodes = np.where(node_name == 1)[0].astype(types.np_int_type)
    outnodes = np.where(node_name == 2)[0].astype(types.np_int_type)
    uppernodes = np.where(node_name == 3)[0].astype(types.np_int_type)
    bottomnodes = np.where(node_name == 4)[0].astype(types.np_int_type)
    halonodes = np.where(node_name == 10)[0].astype(types.np_int_type)
    if self.size == 1:
      halonodes = np.asarray([], dtype=types.np_int_type)

    periodicinnodes = np.where(node_name == 11)[0].astype(types.np_int_type)
    periodicoutnodes = np.where(node_name == 22)[0].astype(types.np_int_type)
    periodicuppernodes = np.where(node_name == 33)[0].astype(types.np_int_type)
    periodicbottomnodes = np.where(node_name == 44)[0].astype(types.np_int_type)

    boundarynodes = np.concatenate([innodes, outnodes, bottomnodes, uppernodes])
    periodicboundarynodes = np.concatenate([periodicinnodes, periodicoutnodes, periodicbottomnodes, periodicuppernodes])

    frontfaces = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    backfaces = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    periodicfrontfaces = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    periodicbackfaces = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    frontnodes = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    backnodes = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    periodicfrontnodes = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    periodicbacknodes = np.zeros(shape=0, dtype=types.np_int_type) # only on 3d
    if self.dim == 3:
      frontfaces = np.where(face_name == 5)[0].astype(types.np_int_type)
      backfaces = np.where(face_name == 6)[0].astype(types.np_int_type)
      periodicfrontfaces = np.where(face_name == 55)[0].astype(types.np_int_type)
      periodicbackfaces = np.where(face_name == 66)[0].astype(types.np_int_type)

      frontnodes = np.where(node_name == 5)[0].astype(types.np_int_type)
      backnodes = np.where(node_name == 6)[0].astype(types.np_int_type)
      periodicfrontnodes = np.where(node_name == 55)[0].astype(types.np_int_type)
      periodicbacknodes = np.where(node_name == 66)[0].astype(types.np_int_type)

      boundaryfaces = np.concatenate([boundaryfaces, backfaces, frontfaces])
      periodicboundaryfaces = np.concatenate([periodicboundaryfaces, periodicbackfaces, periodicfrontfaces])

      boundarynodes = np.concatenate([boundarynodes, backnodes, frontnodes])
      periodicboundarynodes = np.concatenate([periodicboundarynodes, periodicbacknodes, periodicfrontnodes])

    boundaryfaces = np.sort(boundaryfaces)
    periodicboundaryfaces = np.sort(periodicboundaryfaces)
    boundarynodes = np.sort(boundarynodes)
    periodicboundarynodes = np.sort(periodicboundarynodes)

    return (
      innerfaces,
      infaces,
      outfaces,
      upperfaces,
      bottomfaces,
      halofaces,
      periodicinfaces,
      periodicoutfaces,
      periodicupperfaces,
      periodicbottomfaces,
      boundaryfaces,
      periodicboundaryfaces,
      innernodes,
      innodes,
      outnodes,
      uppernodes,
      bottomnodes,
      halonodes,
      periodicinnodes,
      periodicoutnodes,
      periodicuppernodes,
      periodicbottomnodes,
      boundarynodes,
      periodicboundarynodes,
      frontfaces,
      backfaces,
      periodicfrontfaces,
      periodicbackfaces,
      frontnodes,
      backnodes,
      periodicfrontnodes,
      periodicbacknodes,
    )

  def _define_BCs(self, periodicinfaces, periodicupperfaces, periodicfrontfaces):

    BCs = {"in": ["neumann", 1], "out": ["neumann", 2], "upper": ["neumann", 3], "bottom": ["neumann", 4]}

    if len(periodicinfaces) != 0:
      BCs["in"] = ["periodic", 11]
      BCs["out"] = ["periodic", 22]

    if len(periodicupperfaces) != 0:
      BCs["bottom"] = ["periodic", 44]
      BCs["upper"] = ["periodic", 33]

    if self.dim == 3:
      BCs["front"] = ["neumann", 5]
      BCs["back"] = ["neumann", 6]

      if len(periodicfrontfaces) != 0:
        BCs["front"] = ["periodic", 55]
        BCs["back"] = ["periodic", 66]

    return BCs

  def _define_bounds(self, nodes):
    """
    define the boudaries of the geometry
    """
    bounds = None

    if self.dim == 2:
      bounds = np.array([[min(nodes[:, 0]), max(nodes[:, 0])],
                               [min(nodes[:, 1]), max(nodes[:, 1])]],
                              dtype=types.np_float_type)

    if self.dim == 3:
      bounds = np.array([[min(nodes[:, 0]), max(nodes[:, 0])],
                               [min(nodes[:, 1]), max(nodes[:, 1])],
                               [min(nodes[:, 2]), max(nodes[:, 2])]],
                              dtype=types.np_float_type)

    return bounds



