import numpy as np
from manapy.backends.debug import log_step
import manapy.domain.compute as compute
import manapy.domain.utils as utils
from manapy.domain.LocalDomainStructClass import LocalDomainStruct
from mpi4py import MPI
import os

class LocalDomain:

  def __init__(self, local_domain_struct: 'LocalDomainStruct', rank: 'int', size: 'int'):
    if local_domain_struct is None:
      return

    self.rank = rank
    self.size = size
    self.dim = local_domain_struct.dim
    self.float_precision = 'float32' if local_domain_struct.float_precision == 32 else 'float64'
    self.mpi_float_precision = MPI.FLOAT if local_domain_struct.float_precision == 32 else MPI.DOUBLE
    self.nodes = local_domain_struct.nodes.astype(self.float_precision)
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
    self.node_halos = local_domain_struct.node_halos
    self.node_halophyid = local_domain_struct.node_halophyid
    self.phyid_recv = local_domain_struct.phyid_recv
    self.phyid_recv_part_size = local_domain_struct.phyid_recv_part_size
    self.phyid_send = local_domain_struct.phyid_send
    self.halo_halosext = local_domain_struct.halo_halosext
    self.halo_centvol = local_domain_struct.halo_centvol.astype(self.float_precision)
    self.max_cell_nodeid = local_domain_struct.max_cell_nodeid
    self.max_cell_faceid = local_domain_struct.max_cell_faceid
    self.max_face_nodeid = local_domain_struct.max_face_nodeid
    self.max_node_haloid = local_domain_struct.max_node_haloid
    self.max_cell_halonid = local_domain_struct.max_cell_halonid
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))
    self.nb_phy_faces = np.int32(len(self.phy_faces))
    self.test = False # debug attribute

    log_step.log("Prepare communication")
    (
      self.halo_scount,
      self.halo_rcount,
      self.halo_indsend,
      self.halo_comm_ptr
    ) = self.prepare_comm(self.halo_neighsub, self.halo_halosint)
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
      self.phyid_to_faceid
    ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid, self.node_haloid, self.node_oldname)
    log_step.out()


    log_step.log("create_bf_cellid")
    (
      self.ghost_part_size,
      self.bf_cellid
    ) = self._create_bf_cellid(self.phy_faces, self.phyid_recv, self.phyid_recv_part_size, self.node_cellid, self.phyid_to_faceid, self.cell_faceid, self.rank)
    log_step.out()

    log_step.log("_create_shared_ghost_info")
    (self.shared_ghost_info_int, self.shared_ghost_info_flt) = self._create_shared_ghost_info(self.bf_cellid, self.ghost_part_size, self.cell_center, self.cell_faceid, self.cell_loctoglob, self.face_oldname, self.face_normal, self.face_center, self.face_measure, len(self.phyid_recv))
    log_step.out()

    log_step.log("_share_ghost_info_flt and _share_ghost_info_int")
    self._share_ghost_info(self.rank, self.phyid_recv_part_size, self.shared_ghost_info_flt, self.phyid_send)
    self._share_ghost_info(self.rank, self.phyid_recv_part_size, self.shared_ghost_info_int, self.phyid_send)
    log_step.out()

    # print(self.rank, "Aborting...")
    # # Aborting
    # MPI.COMM_WORLD.Abort()

    log_step.log("_create_ghost_tables")
    (
      self.node_ghostid,
      self.cell_ghostnid,
      self.node_ghostcenter,
      self.node_ghostcenter_info,
      self.face_ghostcenter,
      self.node_ghostfaceinfo
    ) = self._create_ghost_tables(self.shared_ghost_info_int, self.shared_ghost_info_flt, self.cells, self.faces, self.cell_faceid, self.ghost_part_size)
    log_step.out()


    log_step.log("_create_halo_ghost_tables")
    (
      self.cell_haloghostnid,
      self.cell_haloghostcenter,
      self.node_haloghostid,
      self.node_haloghostcenter,
      self.node_haloghostcenter_info,
      self.node_haloghostfaceinfo,
      self.halo_sizehaloghost
    ) = self._create_halo_ghost_tables(self.shared_ghost_info_int, self.shared_ghost_info_flt, self.cells, self.node_cellid, self.node_halophyid, self.node_haloid, self.halo_halosext,self.ghost_part_size, self.node_oldname)
    log_step.out()

    ## TODO the use of this tables !?
    self.node_periodicid = np.zeros((self.nb_nodes, 2), dtype=np.int32)
    self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=np.int32)
    self.cell_periodicfid = np.zeros(self.nb_cells, dtype=np.int32)
    self.cell_shift = np.zeros((self.nb_cells, 3), dtype=self.float_precision)

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
    ) = self._face_gradient_info(self.face_cellid, self.faces, self.face_ghostcenter, self.face_name, self.face_normal, self.cell_center, self.halo_centvol, self.face_haloid, self.nodes, self.cell_shift)
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
    ) = self._variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid, self.node_haloghostid, self.node_periodicid, self.nodes, self.node_oldname, self.face_ghostcenter, self.cell_haloghostcenter, self.halo_centvol, self.cell_shift)
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
      comm_ptr = MPI.COMM_WORLD.Create_dist_graph_adjacent([0], [0], sourceweights=None, destweights=None)
      indsend = np.zeros(1, dtype=np.int32)
      scount = np.zeros(1, dtype=np.uint32)
      rcount = np.zeros(1, dtype=np.uint32)
      return scount, rcount, indsend, comm_ptr

    comm_ptr = MPI.COMM_WORLD.Create_dist_graph_adjacent(halo_neighsub[0], halo_neighsub[0], sourceweights=None,
                                                         destweights=None)
    scount = np.zeros(len(halo_neighsub[1]), dtype=np.uint32)
    rcount = np.zeros(len(halo_neighsub[1]), dtype=np.uint32)

    for i in range(len(halo_neighsub[0])):
      scount[i] = halo_neighsub[1][i]

    comm_ptr.Neighbor_alltoallv(scount, rcount)
    indsend = halo_halosint.copy()

    return scount, rcount, indsend, comm_ptr

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
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)
    apprx_nb_faces = nb_cells * max_cell_faceid # TODO ((nb_cells * max_cell_faceid + boundary_faces) / 2)
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    faces_counter = np.zeros(shape=1, dtype=np.int32)

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
    cell_volume = np.zeros(shape=nb_cells, dtype=self.float_precision)
    cell_center = np.zeros(shape=(nb_cells, 3), dtype=self.float_precision) #TODO make it 2d as will as face.center and face.normal
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
    face_measure = np.zeros(shape=nb_faces, dtype=self.float_precision)
    face_center = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
    face_normal = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
    face_tangent = np.zeros(shape=0, dtype=self.float_precision)
    face_binormal = np.zeros(shape=0, dtype=self.float_precision)

    if self.dim == 2:
      compute.compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
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
      cell_halonid = np.zeros(shape=0, dtype=np.int32)
      face_haloid = np.zeros(shape=0, dtype=np.int32)
      node_haloid = np.zeros(shape=(nb_nodes, 1), dtype=np.int32)
    else:
      cell_halonid = np.zeros(shape=(nb_cells, self.max_cell_halonid + 1), dtype=np.int32)
      face_haloid = np.zeros(shape=nb_faces, dtype=np.int32)
      node_haloid = np.zeros(shape=(nb_nodes, self.max_node_haloid + 1), dtype=np.int32)
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
    face_name = np.zeros(shape=faces.shape[0], dtype=np.int32)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=np.int32)
    phyid_to_faceid = np.ones(shape=phy_faces.shape[0], dtype=np.int32) * -1

    node_name = node_oldname.copy()
    node_name[node_haloid[:, -1] != 0] = 10

    node_phyid = utils.create_node_phyid(phy_faces, nb_nodes)

    compute.define_face_name(phy_faces, phy_faces_name, faces, node_phyid, face_haloid, face_oldname, face_name, phyid_to_faceid)

    return (
      face_oldname,
      face_name,
      node_name,
      phyid_to_faceid
    )

  def _create_bf_cellid(self, phy_faces, phyid_recv, phyid_recv_part_size, node_cellid, phyid_to_faceid, cell_faceid, rank):
    """
    bf_cellid needed to create shared_ghost_info correctly
    bf_cellid => [[cell_id, face_index in cell_id]] for every local physical face id
    the order of boundary cells in bf_cellid follows the same order of physical faces in phyid_recv
    phyid_recv store all physical faces need by this partition either its own or of the other partitions
    phyid_recv store physical faces of this partition by its local index and for the other partitions by global index
    """
    ghost_part_size = np.zeros(shape=2, dtype=np.int32)
    compute.get_ghost_part_size(phyid_recv_part_size, rank, ghost_part_size)

    bf_cellid = np.zeros(shape=(len(phy_faces), 2), dtype=np.int32)
    intersect = np.zeros(shape=2, dtype=np.int32)
    start = ghost_part_size[0]
    end = ghost_part_size[0] + ghost_part_size[1]
    compute.create_bf_cellid(phy_faces, phyid_recv, node_cellid, phyid_to_faceid, cell_faceid, intersect, start, end, bf_cellid)

    return (
      ghost_part_size,
      bf_cellid,
    )



  def _create_shared_ghost_info(self, bf_cellid: 'int[:, :]', ghost_part_size: 'int[:]', cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', phyid_recv_size: 'int'):

    shared_ghost_info_data_size_flt = 10 # (ghostcenter_x&y&z, gamma, face_center_x&y&z, face_normal_x&y&z)
    shared_ghost_info_data_size_int = 4 # (cell_id, face index inside the cell, face_oldname, cell global id)
    shared_ghost_info_flt = np.zeros(shape=(phyid_recv_size, shared_ghost_info_data_size_flt), dtype=self.float_precision)
    shared_ghost_info_int = np.zeros(shape=(phyid_recv_size, shared_ghost_info_data_size_int), dtype=np.int32)

    # TODO remove self
    compute.create_ghost_info(bf_cellid, cell_center, cell_faceid, cell_loctoglob, self.faces, self.nodes, face_oldname, face_normal, face_center, face_measure, shared_ghost_info_int, shared_ghost_info_flt, self.dim, ghost_part_size[0])

    return shared_ghost_info_int, shared_ghost_info_flt

  def _create_ghost_tables(self, shared_ghost_info_int: 'int[:, :]', shared_ghost_info_flt: 'float[:, :]', cells: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', ghost_part_size: 'int[:]'):

    start = ghost_part_size[0]
    end = start + ghost_part_size[1]

    node_nb_ghostid = np.zeros(shape=self.nb_nodes, dtype=np.int32)
    compute.get_ghost_tables_size(shared_ghost_info_int, faces, cell_faceid, node_nb_ghostid, start, end)

    max_node_ghost = np.max(node_nb_ghostid)
    node_ghostid = np.zeros(shape=(self.nb_nodes, max_node_ghost + 1), dtype=np.int32)

    # ------------------------------------------------------------------
    #  node_ghostid
    #  node_ghostcenter
    #  face_ghostcenter
    #  node_ghostfaceinfo
    # ------------------------------------------------------------------

    if self.dim == 2:
      node_ghostcenter_data_size = 2  # [ghost_center x.y]
      node_ghostcenter_info_data_size = 3 # [cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 3  # [ghost_center x.y, gamma]
      node_ghostfaceinfo_data_size = 4  # [face_center x.y, face_normal x.y]
      node_ghostcenter = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size), dtype=self.float_precision) * -1
      node_ghostcenter_info = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_info_data_size), dtype=np.int32) * -1
      face_ghostcenter = np.ones(shape=(self.nb_faces, face_ghostcenter_data_size), dtype=self.float_precision) * -1
      node_ghostfaceinfo = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size), dtype=self.float_precision) * -1

      compute.create_ghost_tables_2d(shared_ghost_info_int, shared_ghost_info_flt, faces, cell_faceid, node_ghostid, node_ghostcenter, node_ghostcenter_info, face_ghostcenter, node_ghostfaceinfo, start, end)
    else:
      node_ghostcenter_data_size = 3  # [ghost_center x.y.z]
      node_ghostcenter_info_data_size = 3  # [cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 4  # [ghost_center x.y.z, gamma]
      node_ghostfaceinfo_data_size = 6  # [face_center x.y.z, face_normal x.y.z]
      node_ghostcenter = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size), dtype=self.float_precision) * -1 # like old domain *-1
      node_ghostcenter_info = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_info_data_size), dtype=np.int32) * -1 # like old domain *-1
      face_ghostcenter = np.ones(shape=(self.nb_faces, face_ghostcenter_data_size), dtype=self.float_precision) * -1
      node_ghostfaceinfo = np.ones(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size),
                                    dtype=self.float_precision) * -1 # like old domain *-1

      compute.create_ghost_tables_3d(shared_ghost_info_int, shared_ghost_info_flt, faces, cell_faceid, node_ghostid, node_ghostcenter, node_ghostcenter_info, face_ghostcenter, node_ghostfaceinfo, start, end)

    # ------------------------------------------------------------------
    # cell_ghostnid
    # ------------------------------------------------------------------

    # TODO check why self.nb_faces
    ghost_i_visited = np.ones(shape=self.nb_faces, dtype=np.int32) * -1
    cell_ghostnid_size = np.zeros(shape=self.nb_cells, dtype=np.int32)
    compute.get_cell_ghostnid_size(cells, node_ghostid, ghost_i_visited, cell_ghostnid_size)

    ghost_i_visited.fill(-1)
    max_cell_ghostnid = np.max(cell_ghostnid_size)
    cell_ghostnid = np.zeros(shape=(self.nb_cells, max_cell_ghostnid + 1), dtype=np.int32)

    compute.create_cell_ghostnid(cells, node_ghostid, ghost_i_visited, cell_ghostnid)


    return (
      node_ghostid,
      cell_ghostnid,
      node_ghostcenter,
      node_ghostcenter_info,
      face_ghostcenter,
      node_ghostfaceinfo
    )

  @staticmethod
  def _local_share_ghost_info(local_domains: 'LocalDomain[:]'):
    if len(local_domains) == 1:
      return

    size = len(local_domains)
    recv_data = np.ndarray(shape=(size, size), dtype=object)
    recv_data.fill(None)

    # ------------------------------------------------------------------
    # 1. Send
    # ------------------------------------------------------------------
    for rank in range(size):
      domain = local_domains[rank]

      phyid_send = domain.phyid_send
      shared_ghost_info_flt = domain.shared_ghost_info_flt
      shared_ghost_info_int = domain.shared_ghost_info_int

      i = 0
      while i < len(phyid_send):
        dest_part = phyid_send[i]
        start = i + 2
        end = start + phyid_send[i + 1]
        data_indices = phyid_send[start:end]
        data_flt = shared_ghost_info_flt[data_indices]
        data_int = shared_ghost_info_int[data_indices]
        recv_data[rank][dest_part] = (data_int, data_flt)
        i = end

    # ------------------------------------------------------------------
    # 2. Receive
    # ------------------------------------------------------------------
    for rank in range(size):
      domain = local_domains[rank]

      phyid_recv_part_size = domain.phyid_recv_part_size
      shared_ghost_info_flt = domain.shared_ghost_info_flt
      shared_ghost_info_int = domain.shared_ghost_info_int

      start = 0
      for i in range(0, len(phyid_recv_part_size), 2):
        the_sender = phyid_recv_part_size[i]
        size = phyid_recv_part_size[i + 1]
        if the_sender != rank:
          (data_int, data_flt) = recv_data[the_sender][rank]
          end = start + len(data_flt)
          shared_ghost_info_flt[start:end] = data_flt
          shared_ghost_info_int[start:end] = data_int
        start += size



  def _share_ghost_info(self, rank: 'int', phyid_recv_part_size: 'int[:]', ghost_info_table: 'int[:, :]|float[:, :]', phyid_send: 'int[:]'):
    if self.size == 1:
      return
    comm = MPI.COMM_WORLD

    mpi_data_type = self.mpi_float_precision if np.issubdtype(ghost_info_table.dtype, np.floating) else MPI.INT32_T
    recv_data = []
    reqs = []
    # ------------------------------------------------------------------
    # 1. Post non-blocking receives
    # ------------------------------------------------------------------
    start = 0
    for i in range(0, len(phyid_recv_part_size), 2):
      the_sender = phyid_recv_part_size[i]
      size = phyid_recv_part_size[i + 1]
      if the_sender != rank:
        buffer = np.zeros(shape=(size, ghost_info_table.shape[1]), dtype=ghost_info_table.dtype)
        recv_data.append((start, buffer))
        req = comm.Irecv([buffer, mpi_data_type], source=the_sender, tag=0)
        reqs.append(req)
      start += size


    # ------------------------------------------------------------------
    # 2. Post non-blocking sends
    # ------------------------------------------------------------------
    i = 0
    while i < len(phyid_send):
      dest_part = phyid_send[i] # 0
      size = phyid_send[i + 1] # 1
      start = i + 2 # 2 ... 2 + size
      end = start + size
      data_indices = phyid_send[start:end]
      data = ghost_info_table[data_indices]
      req = comm.Isend([data, self.mpi_float_precision], dest=dest_part, tag=0)
      reqs.append(req)
      i = end


    # ------------------------------------------------------------------
    # 3. Wait for all to complete
    # ------------------------------------------------------------------
    statuses = [MPI.Status() for _ in range(len(reqs))]
    try:
      MPI.Request.Waitall(reqs, statuses)
    except MPI.Exception as e:
      print(f"[Rank {rank}] MPI error during Waitall: {e}")

      for i, status in enumerate(statuses):
        errcode = status.Get_error()
        if errcode != MPI.SUCCESS:
          errmsg = MPI.Get_error_string(errcode)
          print(f"[Rank {rank}] Request {i} failed with: {errmsg}")
      raise RuntimeError("MPI error during Waitall")

    # ------------------------------------------------------------------
    # 4. Copy Data to ghost_info_table
    # ------------------------------------------------------------------
    for item in recv_data:
      start = item[0]
      data = item[1]
      end = start + len(data)
      ghost_info_table[start:end] = data

  def _create_halo_ghost_tables(self, shared_ghost_info_int: 'int32[:, :]', shared_ghost_info_flt: 'float[:, :]', cells: 'int[:, :]', node_cellid: 'int[:, :]', node_halophyid: 'int[:, :]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', ghost_part_size, node_oldname):
    nb_nodes = self.nb_nodes
    nb_cells = self.nb_cells

    if self.size == 1:
      cell_haloghostnid = np.zeros(shape=(0, 1), dtype=np.int32)
      cell_haloghostcenter = np.zeros(shape=(0, 1), dtype=self.float_precision)
      node_haloghostid = np.zeros(shape=(nb_nodes, 1), dtype=np.int32)
      node_haloghostcenter = np.ones(shape=(self.nb_nodes, 1, 3), dtype=self.float_precision) * -1 # TODO !! normally it should be zeros(0,1,1) but functions2d.py and core.py need it as ones(...)
      node_haloghostcenter_info = np.ones(shape=(self.nb_nodes, 1, 3), dtype=np.int32) * -1 # TODO
      node_haloghostfaceinfo = np.zeros(shape=(0, 1, 1), dtype=self.float_precision)

    else:
      shared_ghost_info_size = shared_ghost_info_flt.shape[0]

      # ------------------------------------------------------------------
      # create_b_nodeid
      # ------------------------------------------------------------------

      b_nodeid = np.where(node_oldname != 0)[0].astype(np.int32)


      # ------------------------------------------------------------------
      # create_b_ncellid
      # ------------------------------------------------------------------
      b_visited = np.zeros(shape=len(cells), dtype=np.int8)
      max_b_ncellid = compute.get_max_b_ncellid(b_nodeid, node_cellid, b_visited)
      b_visited.fill(0)
      b_ncellid = np.zeros(shape=max_b_ncellid, dtype=np.int32) # cells that has at least one boundary node
      compute.create_b_ncellid(b_nodeid, node_cellid, b_visited, b_ncellid)

      i_visited = np.ones(shape=shared_ghost_info_size, dtype=np.int32) * -1
      max_bcell_halophyid = compute.count_max_bcell_halophyid(cells, b_ncellid, node_halophyid, i_visited)

      bcell_halophyid = np.zeros(shape=(b_ncellid.shape[0], max_bcell_halophyid + 2), dtype=np.int32) # b_ncellid => halo phy_id
      i_visited.fill(-1)
      compute.create_bcell_halophyid(cells, b_ncellid, node_halophyid, i_visited, bcell_halophyid)

      # ------------------------------------------------------------------
      # ghost_new_index
      # ------------------------------------------------------------------
      ghost_new_index = np.ones(shape=shared_ghost_info_size, dtype=np.int32) * -1
      nb_haloghost = compute.create_ghost_new_index(ghost_part_size, ghost_new_index)

      # ------------------------------------------------------------------
      # create_halo_ghost_tables
      # ------------------------------------------------------------------
      cell_haloghostnid = np.zeros(shape=(nb_cells, max_bcell_halophyid + 1), dtype=np.int32)


      max_nb_haloghost = np.max(node_halophyid[:, -1])
      """
      * cell_haloghostnid [[indices point to cell_haloghostcenter]]
      * node_haloghostid [[indices point to cell_haloghostcenter]]
      """
      if self.dim == 2:
        cell_haloghostcenter_data_size = 3 # [[g_x, g_y, unused g_z]]
        node_haloghostcenter_data_size = 2 # [[[g_x, g_y]]]
        node_haloghostcenter_info_data_size = 3 # [[[(halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
        node_haloghostfaceinfo_data_size = 4 # [[[fc_x, fc_y, fn_x, fn_y]]]
        cell_haloghostcenter = np.zeros(shape=(nb_haloghost, cell_haloghostcenter_data_size), dtype=self.float_precision)
        node_haloghostid = np.zeros(shape=(nb_nodes, max_nb_haloghost + 1), dtype=np.int32)
        node_haloghostcenter = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostcenter_data_size), dtype=self.float_precision) * -1 # like old domain
        node_haloghostcenter_info = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostcenter_info_data_size), dtype=np.int32) * -1
        node_haloghostfaceinfo = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostfaceinfo_data_size), dtype=self.float_precision) * -1 # like old domain

        compute.create_halo_ghost_tables_2d(shared_ghost_info_int, shared_ghost_info_flt, bcell_halophyid, b_nodeid, node_halophyid, node_haloid, halo_halosext, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter, node_haloghostcenter_info, node_haloghostfaceinfo)
      else:
        cell_haloghostcenter_data_size = 3 # [[g_x, g_y, g_z]]
        node_haloghostcenter_data_size = 3 # [[[g_x, g_y, g_z]]]
        node_haloghostcenter_info_data_size = 3 # [[[(halo_cell)index point to halosext, face_old_name, index point to cell_haloghostcenter]]]
        node_haloghostfaceinfo_data_size = 6 # [[[fc_x, fc_y, fc_z, fn_x, fn_y, fn_z]]]
        cell_haloghostcenter = np.zeros(shape=(nb_haloghost, cell_haloghostcenter_data_size), dtype=self.float_precision)
        node_haloghostid = np.zeros(shape=(nb_nodes, max_nb_haloghost + 1), dtype=np.int32)
        node_haloghostcenter = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostcenter_data_size), dtype=self.float_precision) * -1
        node_haloghostcenter_info = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostcenter_info_data_size), dtype=np.int32) * -1
        node_haloghostfaceinfo = np.ones(shape=(nb_nodes, max_nb_haloghost, node_haloghostfaceinfo_data_size),
                                          dtype=self.float_precision) * -1 # like old domain


        compute.create_halo_ghost_tables_3d(shared_ghost_info_int, shared_ghost_info_flt, bcell_halophyid, b_nodeid, node_halophyid, node_haloid, halo_halosext, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter, node_haloghostcenter_info, node_haloghostfaceinfo)

    halo_sizehaloghost = np.sum(node_haloghostid[:, -1]) # Two nodes in the same partition can't have the same haloghostId
    return (
      cell_haloghostnid,
      cell_haloghostcenter,
      node_haloghostid,
      node_haloghostcenter,
      node_haloghostcenter_info,
      node_haloghostfaceinfo,
      halo_sizehaloghost
    )


  def _face_gradient_info(self, face_cellid, faces, face_ghostcenter, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, cell_shift):

    face_air_diamond = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
    face_param1 = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
    face_param2 = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
    face_param3 = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
    face_param4 = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
    face_f1 = np.zeros(shape=(self.nb_faces, self.dim), dtype=self.float_precision)
    face_f2 = np.zeros(shape=(self.nb_faces, self.dim), dtype=self.float_precision)
    face_f3 = np.zeros(shape=(self.nb_faces, self.dim), dtype=self.float_precision)
    face_f4 = np.zeros(shape=(self.nb_faces, self.dim), dtype=self.float_precision)

    if self.dim == 2:
      compute.face_gradient_info_2d(face_cellid, faces, face_ghostcenter, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_param4, face_f1, face_f2, face_f3, face_f4, cell_shift)
    else:
      compute.face_gradient_info_3d(face_cellid, faces, face_ghostcenter, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_f1, face_f2, cell_shift)

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

  def _variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, cell_shift):

    node_R_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_number = np.zeros(self.nb_nodes, dtype=np.int32)

    if self.dim == 2:
      compute.variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      compute.variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

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

    cell_nf = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 3), dtype=self.float_precision)
    compute.create_normal_face_of_cell(cell_center, face_center, cell_faceid, face_normal, cell_nf)
    return cell_nf

  def _dist_ortho_function_2d(self, d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]'):

    face_dist_ortho = np.zeros(shape=self.nb_faces, dtype=self.float_precision) # TODO testing
    if self.dim == 2:
      face_dist_ortho = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
      compute.dist_ortho_function_2d(d_innerfaces, d_boundaryfaces, face_cellid, cell_center, face_center, face_normal, face_dist_ortho)
    return face_dist_ortho

  def _update_boundaries(self, face_name, node_name):

    innerfaces = np.where(face_name == 0)[0].astype(np.int32)
    infaces = np.where(face_name == 1)[0].astype(np.int32)
    outfaces = np.where(face_name == 2)[0].astype(np.int32)
    upperfaces = np.where(face_name == 3)[0].astype(np.int32)
    bottomfaces = np.where(face_name == 4)[0].astype(np.int32)
    halofaces = np.where(face_name == 10)[0].astype(np.int32)
    if self.size == 1:
      halofaces = np.asarray([], dtype=np.int32)

    periodicinfaces = np.where(face_name == 11)[0].astype(np.int32)
    periodicoutfaces = np.where(face_name == 22)[0].astype(np.int32)
    periodicupperfaces = np.where(face_name == 33)[0].astype(np.int32)
    periodicbottomfaces = np.where(face_name == 44)[0].astype(np.int32)

    boundaryfaces = np.concatenate([infaces, outfaces, bottomfaces, upperfaces])
    periodicboundaryfaces = np.concatenate([periodicinfaces, periodicoutfaces, periodicbottomfaces, periodicupperfaces])

    innernodes = np.where(node_name == 0)[0].astype(np.int32)
    innodes = np.where(node_name == 1)[0].astype(np.int32)
    outnodes = np.where(node_name == 2)[0].astype(np.int32)
    uppernodes = np.where(node_name == 3)[0].astype(np.int32)
    bottomnodes = np.where(node_name == 4)[0].astype(np.int32)
    halonodes = np.where(node_name == 10)[0].astype(np.int32)
    if self.size == 1:
      halonodes = np.asarray([], dtype=np.int32)

    periodicinnodes = np.where(node_name == 11)[0].astype(np.int32)
    periodicoutnodes = np.where(node_name == 22)[0].astype(np.int32)
    periodicuppernodes = np.where(node_name == 33)[0].astype(np.int32)
    periodicbottomnodes = np.where(node_name == 44)[0].astype(np.int32)

    boundarynodes = np.concatenate([innodes, outnodes, bottomnodes, uppernodes])
    periodicboundarynodes = np.concatenate([periodicinnodes, periodicoutnodes, periodicbottomnodes, periodicuppernodes])

    frontfaces = np.zeros(shape=0, dtype=np.int32) # only on 3d
    backfaces = np.zeros(shape=0, dtype=np.int32) # only on 3d
    periodicfrontfaces = np.zeros(shape=0, dtype=np.int32) # only on 3d
    periodicbackfaces = np.zeros(shape=0, dtype=np.int32) # only on 3d
    frontnodes = np.zeros(shape=0, dtype=np.int32) # only on 3d
    backnodes = np.zeros(shape=0, dtype=np.int32) # only on 3d
    periodicfrontnodes = np.zeros(shape=0, dtype=np.int32) # only on 3d
    periodicbacknodes = np.zeros(shape=0, dtype=np.int32) # only on 3d
    if self.dim == 3:
      frontfaces = np.where(face_name == 5)[0].astype(np.int32)
      backfaces = np.where(face_name == 6)[0].astype(np.int32)
      periodicfrontfaces = np.where(face_name == 55)[0].astype(np.int32)
      periodicbackfaces = np.where(face_name == 66)[0].astype(np.int32)

      frontnodes = np.where(node_name == 5)[0].astype(np.int32)
      backnodes = np.where(node_name == 6)[0].astype(np.int32)
      periodicfrontnodes = np.where(node_name == 55)[0].astype(np.int32)
      periodicbacknodes = np.where(node_name == 66)[0].astype(np.int32)

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
                              dtype=self.float_precision)

    if self.dim == 3:
      bounds = np.array([[min(nodes[:, 0]), max(nodes[:, 0])],
                               [min(nodes[:, 1]), max(nodes[:, 1])],
                               [min(nodes[:, 2]), max(nodes[:, 2])]],
                              dtype=self.float_precision)

    return bounds

  @staticmethod
  def load_and_create(rank: 'int', size: 'int'):
    folder_name = f"local_domain_{size}"
    file_name = f"mesh{rank}.hdf5"
    path = os.path.join(folder_name, file_name)
    local_domain_struct = LocalDomainStruct.load_hd5(path)
    return LocalDomain(local_domain_struct, rank, size)

