from manapy.domain import LocalDomain, LocalDomainInterface
from manapy.backends.debug import log_step
import time
import numpy as np
import manapy.backends.types as types


class LocalDomain1Cpu(LocalDomain):
  def __int__(self):
    pass

  @staticmethod
  def create_local_domains(local_domain_structs: 'LocalDomainInterface[:]'):

    size = len(local_domain_structs)
    local_domain_objs = [LocalDomain1Cpu(None, None, None) for rank in range(size)]

    # ------------------------------------------------------------------
    # Part 1
    # ------------------------------------------------------------------
    for rank in range(size):
      local_domain_struct = local_domain_structs[rank]
      self = local_domain_objs[rank]

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
      self.nb_nodes = types.np_int_type(len(self.nodes))
      self.nb_cells = types.np_int_type(len(self.cells))
      self.nb_phy_faces = types.np_int_type(len(self.phy_faces))
      self.test = True

      self.start = time.time()


      self.phy_faces_comm = None
      self.halo_comm = None
      log_step.log("Prepare communication")
      (
        self.halo_scount,
        self.halo_rcount,
        self.halo_indsend,
        self.halo_comm_ptr
      ) = (None, None, None, None)
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
        self.face_tangent,  # only in 3D, shape is 0 in 2D
        self.face_binormal  # only in 3D, shape is 0 in 2D
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
      ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid,
                                          self.node_haloid, self.node_oldname)
      log_step.out()


      log_step.log("_create_shared_ghost_info")
      (self.ghost_info_int, self.ghost_info_flt) = self._create_ghost_info(self.cell_center, self.cell_faceid,
                                                                                  self.cell_loctoglob,
                                                                                  self.face_oldname, self.face_normal,
                                                                                  self.face_center, self.face_measure, self.faces,
                                                                                  self.nodes, self.phy_faces,
                                                                                  self.node_cellid,
                                                                                  self.phyid_to_faceid)
      log_step.out()

      log_step.log("_create_ghost_tables")
      (
        self.node_ghostid,
        self.cell_ghostid
      ) = self._create_ghost_tables(self.ghost_info_int, self.node_cellid, self.faces, self.cell_faceid)
      log_step.out()

    # ------------------------------------------------------------------
    # Share
    # ------------------------------------------------------------------
    LocalDomain._local_share_ghost_info(local_domain_objs)
    # ------------------------------------------------------------------
    # Part 2
    # ------------------------------------------------------------------
    for rank in range(size):
      self = local_domain_objs[rank]

      log_step.log("_create_halo_ghost_tables")
      (
        self.cell_haloghostid,
        self.node_haloghostid,
        self.halo_sizehaloghost
      ) = self._create_halo_ghost_tables(self.ext_ghost_info_int, self.node_halophyid, self.cell_halophyid,
                                         self.node_haloid, self.halo_halosext)
      log_step.out()

    # Check if mesh is will constructed
    log_step.log("_check_phy_faces")
    LocalDomain._local_check_phy_faces(local_domain_objs)
    log_step.out()

    # ------------------------------------------------------------------
    # Part 3
    # ------------------------------------------------------------------

    for rank in range(size):
      self = local_domain_objs[rank]

      ## TODO the use of this tables !?
      self.node_periodicid = np.zeros((self.nb_nodes, 2), dtype=types.np_int_type)
      self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=types.np_int_type)
      self.cell_periodicfid = np.zeros(self.nb_cells, dtype=types.np_int_type)
      self.cell_shift = np.zeros((self.nb_cells, 3), dtype=types.np_float_type)

      log_step.log("face_gradient_info")
      (
        self.face_air_diamond,
        self.face_param1,
        self.face_param2,
        self.face_param3,
        self.face_param4,
        self.face_f1,
        self.face_f2,
        self.face_f3,
        self.face_f4  # Only on 2D
      ) = self._face_gradient_info(self.face_cellid, self.faces, self.face_to_phyid, self.ghost_info_flt, self.face_name,
                                   self.face_normal, self.cell_center, self.halo_centvol, self.face_haloid, self.nodes,
                                   self.cell_shift)
      log_step.out()

      log_step.log("_variables")
      (
        self.node_R_x,
        self.node_R_y,
        self.node_R_z,  # Only on 3D
        self.node_lambda_x,
        self.node_lambda_y,
        self.node_lambda_z,  # Only on 3D
        self.node_number,
      ) = self._variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid,
                          self.node_haloghostid, self.node_periodicid, self.nodes, self.node_oldname,
                          self.ghost_info_flt, self.ext_ghost_info_flt, self.halo_centvol, self.cell_shift)
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
        self.frontfaces,  # only in 3D, shape is 0 in 2D
        self.backfaces,  # only in 3D, shape is 0 in 2D
        self.periodicfrontfaces,  # only in 3D, shape is 0 in 2D
        self.periodicbackfaces,  # only in 3D, shape is 0 in 2D
        self.frontnodes,  # only in 3D, shape is 0 in 2D
        self.backnodes,  # only in 3D, shape is 0 in 2D
        self.periodicfrontnodes,  # only in 3D, shape is 0 in 2D
        self.periodicbacknodes,  # only in 3D, shape is 0 in 2D
      ) = self._update_boundaries(self.face_name, self.node_name)
      log_step.out()

      log_step.log("_define_BCs")
      self.BCs = self._define_BCs(self.periodicinfaces, self.periodicupperfaces, self.periodicfrontfaces)
      log_step.out()

      log_step.log("_create_normal_face_of_cell_2d")
      self.cell_nf = self._create_normal_face_of_cell(self.cell_center, self.face_center, self.cell_faceid,
                                                         self.face_normal)
      log_step.out()

      log_step.log("_dist_ortho_function_2d")
      # only in 2D, shape is 0 in 3D
      self.face_dist_ortho = self._dist_ortho_function_2d(self.innerfaces, self.boundaryfaces, self.face_cellid,
                                                          self.cell_center, self.face_center, self.face_normal)
      log_step.out()



    return local_domain_objs
