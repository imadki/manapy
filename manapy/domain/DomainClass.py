import os

from manapy.domain.LocalDomainClass import LocalDomain
from manapy.domain.MeshClass import Mesh
from manapy.domain.PartitioningClass import Partitioning
from manapy.domain.geometry import Cell, Node, Halo, Face, Ghost
import shutil
from mpi4py import MPI
from manapy.domain import VTKWriter
import manapy.backends.types as types
from manapy.domain.LocalDomainInterface import LocalDomainInterface
import manapy.domain.domain_compute as compute
from manapy.backends.debug import log_step


class Domain:
  PartitioningClass = Partitioning
  def __init__(self, local_domain: 'LocalDomain', backend=None):
    from manapy.backends import get_backend
    self.backend = get_backend("cpu") if backend is None else (get_backend(backend) if isinstance(backend, str) else backend)
    # Init
    self.rank = local_domain.rank
    self.size = local_domain.size
    self.dim = local_domain.dim
    self.nbnodes = local_domain.nb_nodes
    self.nbcells = local_domain.nb_cells
    self.nbfaces = local_domain.nb_faces
    self.nbhalos = local_domain.halo_halosext.shape[0]
    self.nbghost = local_domain.nb_phy_faces
    self._maxcellfid = local_domain.max_cell_faceid
    self._maxcellnodeid = local_domain.max_cell_nodeid
    self._maxfacenid = local_domain.max_face_nodeid
    self.test = local_domain.test # debug attribute


    self.comm = MPI.COMM_WORLD


    self.cells = Cell()
    self.nodes = Node()
    self.faces = Face()
    self.halos = Halo()
    self.ghost = Ghost()

    # Cells
    self.cells._nbcells = local_domain.nb_cells
    self.cells._nodeid = local_domain.cells
    self.cells._type = local_domain.cells_type
    self.cells._faceid = local_domain.cell_faceid
    self.cells._cellfid = local_domain.cell_cellfid
    self.cells._cellnid = local_domain.cell_cellnid
    self.cells._halonid = local_domain.cell_halonid
    self.cells._ghostnid = local_domain.cell_ghostid
    self.cells._haloghostnid = local_domain.cell_haloghostid
    self.cells._center = local_domain.cell_center # dimension always 3D
    self.cells._volume = local_domain.cell_volume
    self.cells._nf = local_domain.cell_nf # dimension always 3D
    self.cells._loctoglob = local_domain.cell_loctoglob
    self.cells._tc = local_domain.cell_tc
    self.cells._periodicfid = local_domain.cell_periodicfid
    self.cells._periodicnid = local_domain.cell_periodicnid
    self.cells._shift = local_domain.cell_shift

    # Nodes
    self.nodes._nbnodes = local_domain.nb_nodes
    self.nodes._vertex = local_domain.nodes # in old domain this has dimension (nb_nodes, 4) -> new domain (nb_nodes, 3) 4 is the node_oldname
    self.nodes._name = local_domain.node_name
    self.nodes._oldname = local_domain.node_oldname
    self.nodes._cellid = local_domain.node_cellid
    self.nodes._ghostid = local_domain.node_ghostid
    self.nodes._haloghostid = local_domain.node_haloghostid
    self.nodes._loctoglob = local_domain.node_loctoglob
    self.nodes._halonid = local_domain.node_haloid
    self.nodes._periodicid = local_domain.node_periodicid
    self.nodes._R_x = local_domain.node_R_x
    self.nodes._R_y = local_domain.node_R_y
    self.nodes._R_z = local_domain.node_R_z
    self.nodes._number = local_domain.node_number
    self.nodes._lambda_x = local_domain.node_lambda_x
    self.nodes._lambda_y = local_domain.node_lambda_y
    self.nodes._lambda_z = local_domain.node_lambda_z

    # Faces
    self.faces._nbfaces = local_domain.nb_faces
    self.faces._nodeid = local_domain.faces
    self.faces._cellid = local_domain.face_cellid
    self.faces._name = local_domain.face_name
    self.faces._oldname = local_domain.face_oldname
    self.faces._normal = local_domain.face_normal # always 3D
    self.faces._mesure = local_domain.face_measure
    self.faces._center = local_domain.face_center # always 3D
    self.faces._dist_ortho = local_domain.face_dist_ortho # ?? in old domain in 3D it is useless to define dist_ortho with shape nbfaces TODO
    self.faces._halofid = local_domain.face_haloid
    self.faces._param1 = local_domain.face_param1
    self.faces._param2 = local_domain.face_param2
    self.faces._param3 = local_domain.face_param3
    self.faces._param4 = local_domain.face_param4
    self.faces._f_1 = local_domain.face_f1
    self.faces._f_2 = local_domain.face_f2
    self.faces._f_3 = local_domain.face_f3
    self.faces._f_4 = local_domain.face_f4
    self.faces._airDiamond = local_domain.face_air_diamond
    self.faces._tangent = local_domain.face_tangent
    self.faces._binormal = local_domain.face_binormal
    self.faces._ghost_id = local_domain.face_to_phyid
    self.faces._fv_coeff = local_domain.face_fv_coeff
    self.faces._fv_corrx = local_domain.face_fv_corrx
    self.faces._fv_corry = local_domain.face_fv_corry
    self.faces._fv_corrz = local_domain.face_fv_corrz
    self.faces._fv_weight_left = local_domain.face_fv_weight_left

    # Halos
    self.halos._nb_halos = len(local_domain.halo_centvol)
    self.halos._halosext = local_domain.halo_halosext
    self.halos._neigh = local_domain.halo_neighsub
    self.halos._halosint = local_domain.halo_halosint
    self.halos._centvol = local_domain.halo_centvol
    self.halos._sizehaloghost = len(local_domain.ext_ghost_info_flt)
    self.halo_comm =  local_domain.halo_comm
    self.phy_faces_comm = local_domain.phy_faces_comm

    #Ghost
    self.ghost._nb_ghosts = len(local_domain.ghost_info_int)
    self.ghost._nb_haloghosts = len(local_domain.ext_ghost_info_int)
    self.ghost._info_int = local_domain.ghost_info_int
    self.ghost._info_flt = local_domain.ghost_info_flt
    self.ghost._ext_info_int = local_domain.ext_ghost_info_int
    self.ghost._ext_info_flt = local_domain.ext_ghost_info_flt
    self.ghost._faceid = local_domain.phyid_to_faceid

    # VTK
    vtkprecision = "Float32" if types.FLOAT_TYPE == "float32" else "Float64"
    self.vtk_writer = VTKWriter(self.nodes, self.dim, self.cells.nodeid, self.cells.type, self.comm, vtkprecision)

    # Domain
    self._bounds = local_domain.bounds
    self._BCs = local_domain.BCs
    self._innerfaces = local_domain.innerfaces
    self._infaces = local_domain.infaces
    self._outfaces = local_domain.outfaces
    self._upperfaces = local_domain.upperfaces
    self._bottomfaces = local_domain.bottomfaces
    self._halofaces = local_domain.halofaces
    self._periodicinfaces = local_domain.periodicinfaces
    self._periodicoutfaces = local_domain.periodicoutfaces
    self._periodicupperfaces = local_domain.periodicupperfaces
    self._periodicbottomfaces = local_domain.periodicbottomfaces
    self._boundaryfaces = local_domain.boundaryfaces
    self._periodicboundaryfaces = local_domain.periodicboundaryfaces
    self._innernodes = local_domain.innernodes
    self._innodes = local_domain.innodes
    self._outnodes = local_domain.outnodes
    self._uppernodes = local_domain.uppernodes
    self._bottomnodes = local_domain.bottomnodes
    self._halonodes = local_domain.halonodes
    self._periodicinnodes = local_domain.periodicinnodes
    self._periodicoutnodes = local_domain.periodicoutnodes
    self._periodicuppernodes = local_domain.periodicuppernodes
    self._periodicbottomnodes = local_domain.periodicbottomnodes
    self._boundarynodes = local_domain.boundarynodes
    self._periodicboundarynodes = local_domain.periodicboundarynodes
    self._frontfaces = local_domain.frontfaces
    self._backfaces = local_domain.backfaces
    self._periodicfrontfaces = local_domain.periodicfrontfaces
    self._periodicbackfaces = local_domain.periodicbackfaces
    self._frontnodes = local_domain.frontnodes
    self._backnodes = local_domain.backnodes
    self._periodicfrontnodes = local_domain.periodicfrontnodes
    self._periodicbacknodes = local_domain.periodicbacknodes

    self.BCs = self._BCs
    self.innerfaces = self._innerfaces
    self.infaces = self._infaces
    self.outfaces = self._outfaces
    self.bottomfaces = self._bottomfaces
    self.upperfaces = self._upperfaces
    self.halofaces = self._halofaces
    self.innernodes = self._innernodes
    self.innodes = self._innodes
    self.outnodes = self._outnodes
    self.bottomnodes = self._bottomnodes
    self.uppernodes = self._uppernodes
    self.halonodes = self._halonodes
    self.boundaryfaces = self._boundaryfaces
    self.boundarynodes = self._boundarynodes
    self.periodicboundaryfaces = self._periodicboundaryfaces
    self.periodicboundarynodes = self._periodicboundarynodes

    self.periodicinfaces = self._periodicinfaces
    self.periodicoutfaces = self._periodicoutfaces
    self.periodicupperfaces = self._periodicupperfaces
    self.periodicfrontfaces = self._periodicfrontfaces
    self.frontfaces = self._frontfaces
    self.backfaces = self._backfaces
    self.frontnodes = self._frontnodes
    self.backnodes = self._backnodes
    self.bounds = self._bounds

    if self.backend.name == "gpu":
      self._prepare_gpu_storage()

  def _prepare_gpu_storage(self):
    from manapy.backends.gpu import set_active_backend, GPUArray
    from manapy.comms import GPUNeighborCommunication
    set_active_backend(self.backend)
    GPUArray.convert_to_gpu_array([self, self.cells, self.faces, self.nodes, self.halos, self.ghost])
    self.halo_comm = GPUNeighborCommunication(self.halo_comm, self.backend)


  @staticmethod
  def _all_local_mesh_files_exist(size: int):
    folder_name = f"local_domain_{size}"
    for rank in range(size):
      file_path = os.path.join(folder_name, f"mesh{rank}.hdf5")
      if not os.path.isfile(file_path):
        return False
    return True

  @staticmethod
  def _delete_local_domain_folder(size: int):
    folder_name = f"local_domain_{size}"
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
      shutil.rmtree(folder_name)

  @staticmethod
  def _partitioning_method_name(partitioning_method):
    names = {
      Partitioning.Par_Graph_K_Way: "graph_kway",
      Partitioning.Par_Dual: "mesh_dual",
      Partitioning.Par_Nodal: "mesh_nodal",
    }
    return names.get(partitioning_method, str(partitioning_method))

  @staticmethod
  def _print_run_info(mesh_path, dim, size, partitioning_method, recreate, mesh=None):
    print(f"====> mesh: {mesh}", flush=True)
    print("====> Run info <=====", flush=True)
    print(f"  MPI ranks: {size}", flush=True)
    print(f"  Dimension: {dim}D", flush=True)
    print(f"  Mesh: {mesh_path}", flush=True)
    print(f"  Partitioning: {Domain._partitioning_method_name(partitioning_method)}", flush=True)
    print(f"  Local domains: {'recreate' if recreate else 'reuse if available'}", flush=True)
    print(f"  Precision: {types.INT_TYPE} {types.FLOAT_TYPE}", flush=True)
    if mesh is not None:
      print(f"  Cells: {len(mesh.cells)}", flush=True)
      print(f"  Nodes: {len(mesh.points)}", flush=True)
      print(f"  Faces: {mesh.nb_faces}", flush=True)
      print(f"  Physical faces: {len(mesh.phy_faces)}", flush=True)
    print("====================", flush=True)

  @staticmethod
  def create_domain(mesh_path, dim, partitioning_method=Partitioning.Par_Nodal, recreate=True, backend=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Compile all domain kernels here, once, uniformly on every rank (before the
    # rank-0-only partitioning step). Keeps zero compilation at import while
    # staying barrier-safe.
    log_step.log("domain: compute.setup")
    compute.setup(dim)
    log_step.out("domain: compute.setup")

    if size == 1:
      if recreate == True or not Domain._all_local_mesh_files_exist(size):
        mesh = Mesh(mesh_path, dim, show_info=True)
        Domain._print_run_info(mesh_path, dim, size, partitioning_method, recreate, mesh)
        partitioner = Partitioning(mesh)
        local_domain_data = partitioner.create_sub_domains()
        LocalDomainInterface.save_local_domains(local_domain_data, size)
        local_domain = LocalDomain(local_domain_data[0], rank, size, backend=backend)
        return Domain(local_domain, backend=backend)
      else:
        try:
          local_domain_struct = LocalDomainInterface.load_and_create(rank, size)
          local_domain = LocalDomain(local_domain_struct, rank, size, backend=backend)
          return Domain(local_domain, backend=backend)
        except Exception as e:
          import traceback
          print(f"[Rank {rank}] failed: {e} {traceback.format_exc()}")
          raise
    else:
      log_step.log("domain: rank0 partition/check + barrier")
      if rank == 0:
        try:
          if recreate == True or not Domain._all_local_mesh_files_exist(size):
            print("here====>")
            Domain._delete_local_domain_folder(size)
            mesh = Mesh(mesh_path, dim, show_info=True)
            Domain._print_run_info(mesh_path, dim, size, partitioning_method, recreate, mesh)
            partitioner = Partitioning(mesh)
            partitioner.set_part_vert(size, partitioning_method)
            local_domains = partitioner.create_sub_domains()
            LocalDomainInterface.save_local_domains(local_domains, size)
            print("====> Domain ready <=====", flush=True)
          else:
            Domain._print_run_info(mesh_path, dim, size, partitioning_method, recreate)
        except Exception as e:
          import traceback
          print(f"[Rank 0] failed: {e} {traceback.format_exc()}")
          comm.Abort(1)

      comm.Barrier()
      log_step.out("domain: rank0 partition/check + barrier")


      try:
        log_step.log("domain: load hdf5")
        local_domain_struct = LocalDomainInterface.load_and_create(rank, size)
        log_step.out("domain: load hdf5")
        log_step.log("domain: LocalDomain build")
        local_domain = LocalDomain(local_domain_struct, rank, size, backend=backend)
        log_step.out("domain: LocalDomain build")
      except Exception as e:
        import traceback
        print(f"[Rank {rank}] failed: {e} {traceback.format_exc()}")
        comm.Abort(1)

      log_step.log("domain: final barrier")
      comm.Barrier()
      log_step.out("domain: final barrier")
      log_step.log("domain: Domain wrap")
      d = Domain(local_domain, backend=backend)
      log_step.out("domain: Domain wrap")
      return d


  def save_on_node_multi(self, variables, values, dt=0, time=0, niter=0, miter=0):
    self.vtk_writer.save_node_multi(variables, values, miter, niter, time, dt)

  def save_on_cell_multi(self, variables, values, dt=0, time=0, niter=0, miter=0):
    self.vtk_writer.save_cell_multi(variables, values, miter, niter, time, dt)

