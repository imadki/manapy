import numpy as np
from mpi4py import MPI
from manapy.backends.debug import log_step
from manapy.domain.LocalDomainInterface import LocalDomainInterface
import sys
from manapy.comms.NeighborCommunication import NeighborCommunication
from manapy.compute import DomainCompute

class LocalDomain:

  def __init__(self, local_domain_struct: 'LocalDomainInterface', rank: 'int', size: 'int'):
    self.config = local_domain_struct.config
    self.compute = DomainCompute(self.config)
    self.rank = rank
    self.size = size
    self.dim = local_domain_struct.dim
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
    self.nb_nodes = self.config.int_dtype(len(self.nodes))
    self.nb_cells = self.config.int_dtype(len(self.cells))
    self.nb_phy_faces = self.config.int_dtype(len(self.phy_faces))
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
    self.node_cellid = self.compute.create_node_cellid(self.cells, self.nb_nodes)
    log_step.out()

    log_step.log("cell_cellnid")
    self.cell_cellnid = self.compute.create_cell_cellnid(self.cells, self.node_cellid)
    log_step.out()

    log_step.log("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid
    ) = self.compute.create_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    self.nb_faces = len(self.faces)
    log_step.out()


    log_step.log("Create cell volume and center")
    (
      self.cell_volume,
      self.cell_center
    ) = self.compute.create_cell_info(self.cells, self.nodes, self.dim)
    log_step.out()

    log_step.log("Face measure face center face normal")
    (
      self.face_measure,
      self.face_center,
      self.face_normal,
      self.face_tangent, # only in 3D, shape is 0 in 2D
      self.face_binormal # only in 3D, shape is 0 in 2D
    ) = self.compute.create_face_info(self.faces, self.nodes, self.face_cellid, self.cell_center, self.dim)
    log_step.out()

    log_step.log("cell_halonid, face_haloid, node_haloid")
    (
      self.cell_halonid,
      self.face_haloid,
      self.node_haloid
    ) = self.compute.create_halo_cells(self.cells, self.faces, self.nodes, self.node_halos, self.halo_halosext, self.size, self.max_cell_halonid, self.max_node_haloid)
    log_step.out()

    log_step.log("Create face and node names")
    (
      self.face_oldname,
      self.face_name,
      self.node_name,
      self.phyid_to_faceid,
      self.face_to_phyid,
    ) = self.compute.define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid, self.node_haloid, self.node_oldname, self.nb_nodes)
    log_step.out()


    log_step.log("_create_shared_ghost_info")
    (self.ghost_info_int, self.ghost_info_flt) = self.compute.create_ghost_info(self.cell_center, self.cell_faceid, self.cell_loctoglob, self.face_oldname, self.face_normal, self.face_center, self.face_measure, self.faces, self.nodes, self.phy_faces, self.node_cellid, self.phyid_to_faceid, self.nb_phy_faces, self.phy_faces_name, self.dim)
    log_step.out()

    log_step.log("_share_ghost_info_flt and _share_ghost_info_int")
    (
      self.ext_ghost_info_flt,
      self.ext_ghost_info_int
    ) = self._phy_faces_exchange(self.phy_faces_comm, self.ghost_info_int, self.ghost_info_flt)
    log_step.out()

    log_step.log("_create_ghost_tables")
    (
      self.node_ghostid,
      self.cell_ghostid
    ) = self.compute.create_ghost_tables(self.ghost_info_int, self.node_cellid, self.faces, self.cell_faceid, self.max_node_phyid, self.max_cell_phyid)
    log_step.out()

    log_step.log("_create_halo_ghost_tables")
    (
      self.cell_haloghostid,
      self.node_haloghostid
    ) = self.compute.create_halo_ghost_tables(self.ext_ghost_info_int, self.node_halophyid, self.cell_halophyid, self.node_haloid, self.halo_halosext, self.max_cell_halophyid, self.max_node_halophyid, self.size, self.nb_nodes, self.nb_cells)
    log_step.out()

    # Check if mesh is will constructed
    log_step.log("_check_phy_faces")
    self._check_phy_faces(self.face_cellid, self.face_haloid, self.face_to_phyid)
    log_step.out()

    ## TODO the use of this tables !?
    self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=self.config.int_dtype)
    self.cell_periodicfid = np.zeros(self.nb_cells, dtype=self.config.int_dtype)
    self.cell_shift = np.zeros((self.nb_cells, 3), dtype=self.config.float_dtype)

    # Same-rank periodic pairing: fill cell_shift + face_cellid partners for
    # periodic faces (names 11/22/33/44) BEFORE the geometry kernels consume them.
    log_step.log("_build_periodic_samerank")
    self.node_periodicid = self.compute.build_periodic_samerank(self.nodes, self.node_cellid, self.faces, self.face_name, self.face_center, self.face_cellid, self.cell_shift, self.dim)
    log_step.out()

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
    ) = self.compute.face_gradient_info(self.face_cellid, self.faces, self.face_to_phyid, self.ghost_info_flt, self.face_name, self.face_normal, self.cell_center, self.halo_centvol, self.face_haloid, self.nodes, self.cell_shift, self.dim)
    log_step.out()

    log_step.log("_fv_face_geometry")
    (
      self.face_fv_coeff,
      self.face_fv_corrx,
      self.face_fv_corry,
      self.face_fv_corrz,
      self.face_fv_weight_left,
    ) = self.compute.fv_face_geometry(self.face_cellid, self.face_name, self.face_normal, self.face_center, self.face_haloid, self.cell_center, self.halo_centvol, self.cell_shift)
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
    ) = self.compute.variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid, self.node_haloghostid, self.node_periodicid, self.nodes, self.node_oldname, self.ghost_info_flt, self.ext_ghost_info_flt, self.halo_centvol, self.cell_shift, self.dim)
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
    self.BCs = self._define_BCs()
    log_step.out()

    log_step.log("_create_normal_face_of_cell_2d")
    self.cell_nf = self.compute.create_normal_face_of_cell(self.cell_center, self.face_center, self.cell_faceid, self.face_normal, self.max_cell_faceid)
    log_step.out()

    log_step.log("_dist_ortho_function_2d")
    # only in 2D, shape is 0 in 3D
    self.face_dist_ortho = self.compute.dist_ortho_function_2d(self.innerfaces, self.boundaryfaces, self.face_cellid, self.cell_center, self.face_center, self.face_normal, self.dim)
    log_step.out()



  ##################################################################################3
  ##################################################################################3
  ##################################################################################3
  ##################################################################################3
  ##################################################################################3


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


  def _phy_faces_exchange(self, phy_faces_comm, ghost_info_int: 'int[:, :]', ghost_info_flt: 'float[:, :]'):
    ext_ghost_info_flt = np.zeros(shape=(0, 0), dtype=self.config.float_dtype)
    ext_ghost_info_int = np.zeros(shape=(0, 0), dtype=self.config.int_dtype)
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


  def _check_phy_faces(self, face_cellid, face_haloid, face_to_phyid=None):
    if self.size == 1:
      ext_faces = np.sum(face_cellid[:, 1] == -1)
    else:
      # Real boundary faces: no interior right cell and no halo.
      ext = (face_cellid[:, 1] == -1) & (face_haloid == -1)
      # CROSS-RANK periodic faces became halo faces (face_haloid != -1) yet are
      # still physical faces (face_to_phyid != -1). Count them as boundary faces
      # so the (physical == boundary) balance still holds. Real interface halo
      # faces have face_to_phyid == -1 and are correctly excluded.
      if face_to_phyid is not None:
        ext = ext | ((face_haloid != -1) & (face_to_phyid != -1))
      ext_faces = np.sum(ext)
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
        ext = (ld[i].face_cellid[:, 1] == -1) & (ld[i].face_haloid == -1)
        # cross-rank periodic faces are halo faces that are still physical faces
        ext = ext | ((ld[i].face_haloid != -1) & (ld[i].face_to_phyid != -1))
        ext_faces = np.sum(ext)
      total_ext_faces += ext_faces
      total_phy_faces += len(ld[i].phy_faces)
    if total_ext_faces == 0:
      raise RuntimeError("No physical faces found")
    if total_ext_faces != total_phy_faces:
      raise RuntimeError(f"Mess-constructed mesh number of physical faces are not equal to boundary faces. nb_physical={total_phy_faces} nb_boundary={total_ext_faces}")

  def _update_boundaries(self, face_name, node_name):

    innerfaces = np.where(face_name == 0)[0].astype(self.config.int_dtype)
    infaces = np.where(face_name == 1)[0].astype(self.config.int_dtype)
    outfaces = np.where(face_name == 2)[0].astype(self.config.int_dtype)
    upperfaces = np.where(face_name == 3)[0].astype(self.config.int_dtype)
    bottomfaces = np.where(face_name == 4)[0].astype(self.config.int_dtype)
    halofaces = np.where(face_name == 10)[0].astype(self.config.int_dtype)
    if self.size == 1:
      halofaces = np.asarray([], dtype=self.config.int_dtype)

    periodicinfaces = np.where(face_name == 11)[0].astype(self.config.int_dtype)
    periodicoutfaces = np.where(face_name == 22)[0].astype(self.config.int_dtype)
    periodicupperfaces = np.where(face_name == 33)[0].astype(self.config.int_dtype)
    periodicbottomfaces = np.where(face_name == 44)[0].astype(self.config.int_dtype)

    boundaryfaces = np.concatenate([infaces, outfaces, bottomfaces, upperfaces])
    periodicboundaryfaces = np.concatenate([periodicinfaces, periodicoutfaces, periodicbottomfaces, periodicupperfaces])

    innernodes = np.where(node_name == 0)[0].astype(self.config.int_dtype)
    innodes = np.where(node_name == 1)[0].astype(self.config.int_dtype)
    outnodes = np.where(node_name == 2)[0].astype(self.config.int_dtype)
    uppernodes = np.where(node_name == 3)[0].astype(self.config.int_dtype)
    bottomnodes = np.where(node_name == 4)[0].astype(self.config.int_dtype)
    halonodes = np.where(node_name == 10)[0].astype(self.config.int_dtype)
    if self.size == 1:
      halonodes = np.asarray([], dtype=self.config.int_dtype)

    periodicinnodes = np.where(node_name == 11)[0].astype(self.config.int_dtype)
    periodicoutnodes = np.where(node_name == 22)[0].astype(self.config.int_dtype)
    periodicuppernodes = np.where(node_name == 33)[0].astype(self.config.int_dtype)
    periodicbottomnodes = np.where(node_name == 44)[0].astype(self.config.int_dtype)

    boundarynodes = np.concatenate([innodes, outnodes, bottomnodes, uppernodes])
    periodicboundarynodes = np.concatenate([periodicinnodes, periodicoutnodes, periodicbottomnodes, periodicuppernodes])

    frontfaces = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    backfaces = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    periodicfrontfaces = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    periodicbackfaces = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    frontnodes = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    backnodes = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    periodicfrontnodes = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    periodicbacknodes = np.zeros(shape=0, dtype=self.config.int_dtype) # only on 3d
    if self.dim == 3:
      frontfaces = np.where(face_name == 5)[0].astype(self.config.int_dtype)
      backfaces = np.where(face_name == 6)[0].astype(self.config.int_dtype)
      periodicfrontfaces = np.where(face_name == 55)[0].astype(self.config.int_dtype)
      periodicbackfaces = np.where(face_name == 66)[0].astype(self.config.int_dtype)

      frontnodes = np.where(node_name == 5)[0].astype(self.config.int_dtype)
      backnodes = np.where(node_name == 6)[0].astype(self.config.int_dtype)
      periodicfrontnodes = np.where(node_name == 55)[0].astype(self.config.int_dtype)
      periodicbacknodes = np.where(node_name == 66)[0].astype(self.config.int_dtype)

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

  def _define_BCs(self):

    BCs = {"in": ["neumann", 1], "out": ["neumann", 2], "upper": ["neumann", 3], "bottom": ["neumann", 4]}

    # Detect periodic boundaries GLOBALLY. With cross-rank periodic support, a
    # periodic face may have been turned into a halo face (name 10) on this rank,
    # so the per-rank periodic*faces lists can be empty even though the boundary
    # IS periodic. The physical-face TAG survives the halo relabel, so use the
    # local physical-face names and OR them across ranks so every rank agrees
    # (the Variable BC consistency check requires a domain-wide answer).
    pfn = self.phy_faces_name
    has_inout = bool(np.any(pfn == 11) or np.any(pfn == 22)) if pfn.shape[0] else False
    has_updown = bool(np.any(pfn == 33) or np.any(pfn == 44)) if pfn.shape[0] else False
    has_frontback = bool(np.any(pfn == 55) or np.any(pfn == 66)) if pfn.shape[0] else False
    if self.size > 1:
      has_inout = MPI.COMM_WORLD.allreduce(has_inout, op=MPI.LOR)
      has_updown = MPI.COMM_WORLD.allreduce(has_updown, op=MPI.LOR)
      if self.dim == 3:
        has_frontback = MPI.COMM_WORLD.allreduce(has_frontback, op=MPI.LOR)

    if has_inout:
      BCs["in"] = ["periodic", 11]
      BCs["out"] = ["periodic", 22]

    if has_updown:
      BCs["bottom"] = ["periodic", 44]
      BCs["upper"] = ["periodic", 33]

    if self.dim == 3:
      BCs["front"] = ["neumann", 5]
      BCs["back"] = ["neumann", 6]

      if has_frontback:
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
                              dtype=self.config.float_dtype)

    if self.dim == 3:
      bounds = np.array([[min(nodes[:, 0]), max(nodes[:, 0])],
                               [min(nodes[:, 1]), max(nodes[:, 1])],
                               [min(nodes[:, 2]), max(nodes[:, 2])]],
                              dtype=self.config.float_dtype)

    return bounds



