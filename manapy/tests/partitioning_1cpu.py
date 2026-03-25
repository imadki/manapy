import os
import h5py
from manapy.domain import LocalDomain, LocalDomainStruct
from manapy.backends.debug import log_step
import time
import numpy as np
import subprocess


class SingleCoreDomainTables:
  def __init__(self, local_domains, float_precision):


    self.nb_partitions = len(local_domains)
    self.float_precision = float_precision

    self.d_cells = []
    self.d_faces = []
    self.d_nodes = []
    self.d_cell_nodeid = []
    self.d_cell_faces = []
    self.d_cell_center = []
    self.d_cell_volume = []
    self.d_cell_halonid = []
    self.d_cell_loctoglob = []
    self.d_cell_cellfid = []
    self.d_cell_cellnid = []
    self.d_cell_nf = []
    self.d_cell_ghostnid = []
    self.d_cell_haloghostnid = []
    self.d_cell_haloghostcenter = []
    self.d_node_loctoglob = []
    self.d_node_cellid = []
    self.d_node_name = []
    self.d_node_oldname = []
    self.d_node_ghostid = []
    self.d_node_haloghostid = []
    self.d_node_ghostcenter = []
    self.d_node_ghostcenter_info = []
    self.d_node_haloghostcenter = []
    self.d_node_haloghostcenter_info = []
    self.d_node_ghostfaceinfo = []
    self.d_node_haloghostfaceinfo = []
    self.d_node_halonid = []
    self.d_halo_halosext = []
    self.d_halo_halosint = []
    self.d_halo_neigh = []
    self.d_halo_centvol = []
    self.d_halo_sizehaloghost = []
    self.d_face_halofid = []
    self.d_face_name = []
    self.d_face_normal = []
    self.d_face_center = []
    self.d_face_measure = []
    self.d_face_ghostcenter = []
    self.d_face_oldname = []
    self.d_face_cellid = []
    self.d_node_periodicid = []

    for i in range(len(local_domains)):
      domain = local_domains[i]

      self.d_cells.append(domain.cells.nodeid)
      self.d_faces.append(domain.faces.nodeid)
      self.d_nodes.append(domain.nodes.vertex)
      self.d_cell_nodeid.append(domain.cells.nodeid)
      self.d_cell_faces.append(domain.cells.faceid)
      self.d_cell_center.append(domain.cells.center)
      self.d_cell_volume.append(domain.cells.volume)
      self.d_cell_halonid.append(domain.cells.halonid)
      self.d_cell_loctoglob.append(domain.cells.loctoglob)
      self.d_cell_cellfid.append(domain.cells.cellfid)
      self.d_cell_cellnid.append(domain.cells.cellnid)
      self.d_cell_nf.append(domain.cells.nf)
      self.d_cell_ghostnid.append(domain.cells.ghostnid)
      self.d_cell_haloghostnid.append(domain.cells.haloghostnid)
      self.d_cell_haloghostcenter.append(domain.cells.haloghostcenter)
      self.d_node_loctoglob.append(domain.nodes.loctoglob)
      self.d_node_cellid.append(domain.nodes.cellid)
      self.d_node_name.append(domain.nodes.name)
      self.d_node_oldname.append(domain.nodes.oldname)
      self.d_node_ghostid.append(domain.nodes.ghostid)
      self.d_node_haloghostid.append(domain.nodes.haloghostid)
      self.d_node_ghostcenter.append(domain.nodes.ghostcenter)
      self.d_node_ghostcenter_info.append(domain.nodes.ghostcenter_info)
      self.d_node_haloghostcenter.append(domain.nodes.haloghostcenter)
      self.d_node_haloghostcenter_info.append(domain.nodes.haloghostcenter_info)
      self.d_node_ghostfaceinfo.append(domain.nodes.ghostfaceinfo)
      self.d_node_haloghostfaceinfo.append(domain.nodes.haloghostfaceinfo)
      self.d_node_halonid.append(domain.nodes.halonid)
      self.d_halo_halosext.append(domain.halos.halosext)
      self.d_halo_halosint.append(domain.halos.halosint)
      self.d_halo_neigh.append(domain.halos.neigh)
      self.d_halo_centvol.append(domain.halos.centvol)
      self.d_halo_sizehaloghost.append(domain.halos.sizehaloghost)
      self.d_face_halofid.append(domain.faces.halofid)
      self.d_face_name.append(domain.faces.name)
      self.d_face_normal.append(domain.faces.normal)
      self.d_face_center.append(domain.faces.center)
      self.d_face_measure.append(domain.faces.mesure)
      self.d_face_ghostcenter.append(domain.faces.ghostcenter)
      self.d_face_oldname.append(domain.faces.oldname)
      self.d_face_cellid.append(domain.faces.cellid)
      self.d_node_periodicid.append(domain.nodes.periodicid)

class DomainTables:
  __slots__ = [
    "d_cells",
    "d_faces",
    "d_nodes",
    "d_cell_nodeid",
    "d_cell_faces",
    "d_cell_center",
    "d_cell_volume",
    "d_cell_halonid",
    "d_cell_loctoglob",
    "d_cell_cellfid",
    "d_cell_cellnid",
    "d_cell_nf",
    "d_cell_ghostnid",
    "d_cell_haloghostnid",
    "d_cell_haloghostcenter",
    # "d_cell_tc",
    "d_node_loctoglob",
    "d_node_cellid",
    "d_node_name",
    "d_node_oldname",
    "d_node_ghostid",
    "d_node_haloghostid",
    "d_node_ghostcenter",
    "d_node_ghostcenter_info",
    "d_node_haloghostcenter",
    "d_node_haloghostcenter_info",
    "d_node_ghostfaceinfo",
    "d_node_haloghostfaceinfo",
    "d_node_halonid",
    "d_halo_halosext",
    "d_halo_halosint",
    "d_halo_neigh",
    "d_halo_centvol",
    "d_halo_sizehaloghost",
    # "d_halo_indsend",
    "d_face_halofid",
    "d_face_name",
    "d_face_normal",
    "d_face_center",
    "d_face_measure",
    "d_face_ghostcenter",
    "d_face_oldname",
    "d_face_cellid",
    "nb_partitions",
    "float_precision",
    "d_node_periodicid"
  ]

  def __init__(self, nb_partitions, mesh_name, float_precision, dim, create_par_fun):
    if create_par_fun:
      create_par_fun(nb_partitions, mesh_name, float_precision=float_precision, dim=dim)
    else:
      mpi_exec = "/usr/bin/mpirun"
      python_exec = "/home/aben-ham/anaconda3/envs/work/bin/python3"

      root_file = os.getcwd()
      mesh_file_path = os.path.join(root_file, 'meshes', mesh_name)
      script_path = os.path.join(root_file, 'helpers', 'create_partitions_mpi_worker.py')
      cmd = [mpi_exec, "-n", str(nb_partitions), "--oversubscribe", python_exec, script_path, mesh_file_path,
             float_precision, str(dim)]

      result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
      if result.returncode != 0:
        print(result.__str__(), os.getcwd())
        raise SystemExit(result.returncode)

    self.nb_partitions = nb_partitions
    self.float_precision = float_precision

    for i in range(nb_partitions):
      mesh_dir = "domain_meshes" + str(nb_partitions) + "PROC"
      filename = os.path.join(mesh_dir, f"mesh{i}.hdf5")
      with h5py.File(filename, "r") as f:
        for key in f.keys():
          arr = self.add_attribute_if_not_exists(key, nb_partitions)
          arr[i] = f[key][...]

  def add_attribute_if_not_exists(self, attr_name, nb_partitions):
    if not hasattr(self, attr_name):
      setattr(self, attr_name, [i for i in range(nb_partitions)])
    return getattr(self, attr_name)

class LocalDomain1Cpu(LocalDomain):
  def __int__(self):
    pass

  @staticmethod
  def create_local_domains(local_domain_structs: 'LocalDomainStruct[:]'):

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
      self.float_precision = 'float32' if local_domain_struct.float_precision == 32 else 'float64'
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
      self.halo_halosext = local_domain_struct.halo_halosext
      self.halo_centvol = local_domain_struct.halo_centvol.astype(self.float_precision)
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
      self.nb_nodes = np.int32(len(self.nodes))
      self.nb_cells = np.int32(len(self.cells))
      self.nb_phy_faces = np.int32(len(self.phy_faces))
      self.test = True

      self.start = time.time()


      self.phy_faces_comm = None
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
        self.phyid_to_faceid
      ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid,
                                          self.node_haloid, self.node_oldname)
      log_step.out()


      log_step.log("_create_shared_ghost_info")
      (self.ghost_info_int, self.ghost_info_flt) = self._create_shared_ghost_info(self.cell_center, self.cell_faceid,
                                                                                  self.cell_loctoglob,
                                                                                  self.face_oldname, self.face_normal,
                                                                                  self.face_center, self.face_measure,
                                                                                  self.phyid_send, self.faces,
                                                                                  self.nodes, self.phy_faces,
                                                                                  self.node_cellid,
                                                                                  self.phyid_to_faceid)
      log_step.out()


      log_step.log("_create_ghost_tables")
      (
        self.node_ghostid,
        self.cell_ghostnid,
        self.node_ghostcenter,
        self.node_ghostcenter_info,
        self.face_ghostcenter,
        self.node_ghostfaceinfo
      ) = self._create_ghost_tables(self.ghost_info_int, self.ghost_info_flt, self.cells, self.faces, self.cell_faceid)
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
        self.cell_haloghostnid,
        self.cell_haloghostcenter,
        self.node_haloghostid,
        self.node_haloghostcenter,
        self.node_haloghostcenter_info,
        self.node_haloghostfaceinfo,
        self.halo_sizehaloghost
      ) = self._create_halo_ghost_tables(self.ext_ghost_info_flt, self.ext_ghost_info_int, self.node_halophyid,
                                         self.cell_halophyid, self.node_haloid, self.halo_halosext)
      log_step.out()

      ## TODO the use of this tables !?
      self.node_periodicid = np.zeros((self.nb_nodes, 2), dtype=np.int32)
      self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=np.int32)
      self.cell_periodicfid = np.zeros(self.nb_cells, dtype=np.int32)
      self.cell_shift = np.zeros((self.nb_cells, 3), dtype=self.float_precision)

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
      ) = self._face_gradient_info(self.face_cellid, self.faces, self.face_ghostcenter, self.face_name,
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
                          self.face_ghostcenter, self.cell_haloghostcenter, self.halo_centvol, self.cell_shift)
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
