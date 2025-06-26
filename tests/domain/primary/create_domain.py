import numpy as np
import meshio
import time
import warnings
import manapy_domain
from mpi4py import MPI
from create_domain_utils import (append,
                                 append_1d,
                                 count_max_node_cellid,
                                 create_node_cellid,
                                 count_max_cell_cellnid,
                                 create_cell_cellnid,
                                 create_info,
                                 compute_cell_center_volume_3d,
                                 get_max_node_faceid,
                                 get_node_faceid,
                                 define_face_and_node_name,
                                 compute_cell_center_volume_2d,
                                 compute_face_info_2d,
                                 compute_face_info_3d,
                                 create_cellfid_and_bf_info,
                                 create_halo_cells,
                                 create_ghost_info_2d,
                                 create_ghost_info_3d,
                                 create_ghost_tables_2d,
                                 create_ghost_tables_3d,
                                 get_ghost_part_size,
                                 get_ghost_tables_size,
                                 count_max_bcell_halobfid,
                                 create_bcell_halobfid,
                                 count_max_bf_nodeid,
                                 create_bf_nodeid,
                                 create_ghost_new_index,
                                 create_halo_ghost_tables_2d,
                                 create_halo_ghost_tables_3d,
                                 face_gradient_info_2d,
                                 face_gradient_info_3d,
                                 variables_2d,
                                 variables_3d,
                                 create_normal_face_of_cell_2d,
                                 dist_ortho_function_2d
                                 )

# TODO check indexing limit on int32

class Mesh:
  def __init__(self, mesh_path, dim):
    if not (isinstance(dim, int) and dim == 2 or dim == 3):
      raise ValueError('Invalid dimension')

    mesh, cells_dict, points, cell_data_dict = self._read_mesh(mesh_path)
    cells, cells_type, max_cell_nodeid, max_cell_faceid, max_face_nodeid = self._create_cells(cells_dict, dim)
    phy_faces, phy_faces_name = self._create_phy_faces(cells_dict, cell_data_dict, dim)

    self.mesh = mesh
    self.cells = cells
    self.cells_type = cells_type
    self.max_cell_nodeid = max_cell_nodeid
    self.max_cell_faceid = max_cell_faceid
    self.max_face_nodeid = max_face_nodeid
    self.points = points
    self.phy_faces = phy_faces
    self.phy_faces_name = phy_faces_name
    self.dim = dim

    if len(cells) == 0 or len(points) == 0:
      raise ValueError('Empty mesh')

  def _read_mesh(self, mesh_path):
    mesh = meshio.read(mesh_path)
    MESHIO_VERSION = int(meshio.__version__.split(".")[0])
    if MESHIO_VERSION < 4:
      # print(mesh.cell_data['triangle']['gmsh:physical'])
      # print(mesh.cells['triangle'])
      # need to reverse order of access for compatibility
      cell_data_dict = {}
      for k1 in mesh.cell_data.keys():
        for k2 in mesh.cell_data[k1].keys():
          if cell_data_dict.get(k2) is None:
            cell_data_dict[k2] = {}
          cell_data_dict[k2][k1] = mesh.cell_data[k1][k2]
          print(f"{k2} => {k1}")
      cells_dict = mesh.cells
      # raise NotImplementedError
    else:
      # print(mesh.cell_data_dict['gmsh:physical']['triangle'])
      # print(mesh.cells_dict['triangle'])
      cells_dict = mesh.cells_dict
      cell_data_dict = mesh.cell_data_dict
    points = mesh.points

    return mesh, cells_dict, points, cell_data_dict

  def _create_phy_faces(self, cells_dict, cell_data_dict, dim):
    physicals = cell_data_dict['gmsh:physical']
    physicals_key = ['line']
    if dim == 3:
      physicals_key = ['quad', 'triangle']
    max_nb_face_nodes = 2
    counter = 0

    for k in physicals_key:
      if physicals.get(k) is not None:
        counter += len(physicals[k])
        if k == 'triangle':
          max_nb_face_nodes = max(max_nb_face_nodes, 3)
        elif k == 'quad':
          max_nb_face_nodes = max(max_nb_face_nodes, 4)

    phy_faces = np.zeros(shape=(counter, max_nb_face_nodes + 1), dtype=np.int32)
    phy_faces_name = np.zeros(shape=counter, dtype=np.int32)

    counter = 0
    for k in physicals_key:
      if physicals.get(k) is not None:
        append(phy_faces, cells_dict[k], counter)
        append_1d(phy_faces_name, physicals[k], counter)
        counter += len(physicals[k])

    return phy_faces, phy_faces_name

  def _create_cells(self, meshio_mesh_dic, dim):
    # TODO make cell Types global constant
    allowed_cells = ['quad', 'triangle']
    if dim == 3:
      allowed_cells = ['pyramid', 'hexahedron', 'tetra']
    cell_type_dic = {
      "triangle": 1,
      "quad": 2,
      "tetra": 3,
      "hexahedron": 4,
      "pyramid": 5,
    }
    max_cell_nodeid = -1
    max_cell_faceid = -1
    max_face_nodeid = -1
    for item in meshio_mesh_dic.keys():
      if item == 'triangle':
        max_cell_nodeid = max(max_cell_nodeid, 3)
        max_cell_faceid = max(max_cell_faceid, 3)
        max_face_nodeid = max(max_face_nodeid, 2)
      elif item == 'quad':
        max_cell_nodeid = max(max_cell_nodeid, 4)
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 2)
      elif item == 'tetra':
        max_cell_nodeid = max(max_cell_nodeid, 4)
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 3)
      elif item == 'hexahedron':
        max_cell_nodeid = max(max_cell_nodeid, 8)
        max_cell_faceid = max(max_cell_faceid, 6)
        max_face_nodeid = max(max_face_nodeid, 4)
      elif item == 'pyramid':
        max_cell_nodeid = max(max_cell_nodeid, 5)
        max_cell_faceid = max(max_cell_faceid, 5)
        max_face_nodeid = max(max_face_nodeid, 4)

    number_of_cells = 0
    for item in allowed_cells:
      if meshio_mesh_dic.get(item) is not None:
        number_of_cells += len(meshio_mesh_dic[item])

    cells = np.zeros(shape=(number_of_cells, max_cell_nodeid + 1), dtype=np.int32)
    cells_type = np.zeros(shape=number_of_cells, dtype=np.uint8)

    counter = 0
    for item in allowed_cells:
      if meshio_mesh_dic.get(item) is not None:
        cells_item = meshio_mesh_dic[item]
        cells_type[counter:counter + len(cells_item)] = cell_type_dic[item]
        append(cells, cells_item, counter)
        counter += len(cells_item)

    return cells, cells_type, max_cell_nodeid, max_cell_faceid, max_face_nodeid

class LocalDomainStruct:
  # TODO create cellfid for partitioning with customization
  def __init__(self):
    self.nodes = None # [[node x, y, z]]
    self.cells = None # [[cells nodes]]
    self.cells_type = None # [cell type]
    self.phy_faces = None # [[physical face nodes]]
    self.phy_faces_name = None # [physical face name]
    self.phy_faces_loctoglob = None
    self.bf_cellid = None # [[cellid, face index in the cell]] for each boundary face
    self.cell_loctoglob = None # [cell global index]
    self.node_loctoglob = None # [node global index]
    self.halo_neighsub = None # [sub domain id]
    self.node_halos = None # [node1, number of halos, halocell index in halo_halosext, node2, number of halos, ....] shape=(2*nb_nodes + nb_halos)
    self.node_halobfid = None # [[index0 point to halo_halobf, index1 ..., size]] shape=(nb_nodes, max_node_halobf + 1)
    self.shared_bf_recv = None # [boundary faces global index, ...] description="represent the global index of boundary faces that is needed from this partition either from itself or the other paritions, all other tables that will use boundary faces must point to this table"
    self.bf_recv_part_size = None # [boundary faces part, size]
    self.shared_bf_send = None # [recv_part_index, size, size indices point to shared_bf_recv, ...] description="used when this part need to send its boundary faces to recv_part"
    self.halo_halosext = None # [global_index_if_halo_cell, ...] shape=(nb_halos)
    self.dim = None
    self.float_precision = None
    self.max_cell_nodeid = None
    self.max_cell_faceid = None
    self.max_face_nodeid = None
    self.max_node_haloid = None
    self.max_cell_halofid = None
    self.max_cell_halonid = None



    # To be copied if one partition is specified (recalculated on subdomains)
    self.node_cellid = None
    self.cell_cellnid = None

    ## Temporarily
    self.max_phy_face_nodeid = None
    self.map_cells = {}
    self.map_nodes = {}
    self.map_phy_faces = {}

class Domain:

  def __init__(self, mesh: 'Mesh', float_precision: 'str'):
    if float_precision != 'float32' and float_precision != 'float64':
      raise ValueError('Invalid float precision argument')

    self.nodes = np.array(mesh.points[:, 0:mesh.dim]).astype(np.float32)
    self.cells = mesh.cells
    self.cells_type = mesh.cells_type
    self.max_cell_nodeid = mesh.max_cell_nodeid
    self.max_cell_faceid = mesh.max_cell_faceid
    self.max_face_nodeid = mesh.max_face_nodeid
    self.phy_faces = mesh.phy_faces
    self.phy_faces_name = mesh.phy_faces_name
    self.dim = mesh.dim
    self.float_precision = float_precision
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))
    self.nb_phy_faces = np.int32(len(self.phy_faces))


    self.start = time.time()
    print("node_cellid")
    self.node_cellid = self.create_node_cellid(self.cells, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    self.cell_cellnid = self.create_cell_cellnid(self.cells, self.node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellfid and boundary_faces")
    (
      self.cell_cellfid,
      self.bf_cellid,
      self.bf_nodes
    ) = self._create_cellfid_bf_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid, self.nb_phy_faces)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("node_bfid") # node_boundary_face_id
    self.node_bfid = self.create_node_bfid(self.bf_nodes, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

  # ###############################
  # ###############################

  @staticmethod
  def create_node_cellid(cells: 'int[:, :]', nb_nodes: 'int'):
    # Count max node cellid
    res = np.zeros(shape=nb_nodes, dtype=np.int32)
    count_max_node_cellid(cells, res)
    max_node_cellid = np.max(res)

    # Create node cellid
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32)
    create_node_cellid(cells, node_cellid)
    return node_cellid

  @staticmethod
  def create_node_bfid(bf_nodes: 'int[:, :]', nb_nodes: 'int'):
    # Count max node boundary faces
    # Create node boundary faceid
    return Domain.create_node_cellid(bf_nodes, nb_nodes)



  @staticmethod
  def create_cell_cellnid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    # Count max cell cellnid
    visited = np.zeros(cells.shape[0], dtype=np.int32)
    max_cell_cellnid = count_max_cell_cellnid(cells, node_cellid, visited)

    # Create cell cellnid
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=np.int32)
    create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  @staticmethod
  def _create_cellfid_bf_info(
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int',
    nb_phy_faces: 'int'
  ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    bf_cellid = np.zeros(shape=(nb_phy_faces, 2), dtype=np.int32)
    bf_nodes = np.zeros(shape=(nb_phy_faces, max_face_nodeid + 1), dtype=np.int32)

    create_cellfid_and_bf_info(
      cells,
      node_cellid,
      cell_type,
      tmp_cell_faces,
      tmp_size_info,
      cell_cellfid,
      bf_cellid,
      bf_nodes
    )



    return (cell_cellfid, bf_cellid, bf_nodes)

  def c_create_sub_domains(self, nb_parts: 'int'):
    if nb_parts == 1:
      local_domain = LocalDomainStruct()

      local_domain.nodes = self.nodes
      local_domain.cells = self.cells
      local_domain.cells_type = self.cells_type
      local_domain.phy_faces = self.phy_faces
      local_domain.phy_faces_name = self.phy_faces_name
      local_domain.cell_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.node_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.halo_neighsub = []
      local_domain.node_halos = []
      local_domain.dim = self.dim
      local_domain.float_precision = self.float_precision
      local_domain.max_cell_nodeid = self.max_cell_nodeid
      local_domain.max_cell_faceid = self.max_cell_faceid
      local_domain.max_face_nodeid = self.max_face_nodeid
      local_domain.node_cellid = self.node_cellid
      local_domain.cell_cellnid = self.cell_cellnid

      return [local_domain]

    print("Start creating sub domains")
    # print(self.cell_cellfid)
    res = manapy_domain.create_sub_domains(
      self.cell_cellfid,
      self.node_cellid,
      self.node_bfid,
      self.bf_cellid,
      self.cells,
      self.cell_cellfid,
      self.cell_cellnid,
      self.cells_type,
      self.nodes,
      self.phy_faces,
      self.phy_faces_name,
      nb_parts,
    )

    local_domains = []
    for item in res:
      l_domain = LocalDomainStruct()
      l_domain.nodes = item[0]
      l_domain.cells = item[1]
      l_domain.cells_type = item[2]
      l_domain.phy_faces = item[3]
      l_domain.phy_faces_name = item[4]
      l_domain.phy_faces_loctoglob = item[5]
      l_domain.bf_cellid = item[6]
      l_domain.cell_loctoglob = item[7]
      l_domain.node_loctoglob = item[8]
      l_domain.halo_neighsub = item[9]
      l_domain.node_halos = item[10]
      l_domain.node_halobfid = item[11]
      l_domain.shared_bf_recv = item[12]
      l_domain.bf_recv_part_size = item[13]
      l_domain.shared_bf_send = item[14]
      l_domain.halo_halosext = item[15]
      l_domain.dim = self.dim
      l_domain.float_precision = self.float_precision
      l_domain.max_cell_nodeid = item[16]
      l_domain.max_cell_faceid = item[17]
      l_domain.max_face_nodeid = item[18]
      l_domain.max_node_haloid = item[19]
      l_domain.max_cell_halofid = item[20]
      l_domain.max_cell_halonid = item[21]
      l_domain.node_cellid = None
      l_domain.cell_cellnid = None
      local_domains.append(l_domain)

    print(f"Execution time: {time.time() - self.start:.6f} seconds")
    return local_domains

# TODO: zeos unstead of ones
# TODO: face_haloid, node_haloid rename
# TODO: haloext -> [[cellgid, cellnode1, cellnode2, .., size]] shape=(nb_haloext, max_cell_node + 2)
class LocalDomain:
  def __init__(self, local_domain_struct: 'LocalDomainStruct', rank: 'int', size: 'int'):
    self.rank = rank
    self.size = size
    self.nodes = local_domain_struct.nodes
    self.cells = local_domain_struct.cells
    self.cells_type = local_domain_struct.cells_type
    self.phy_faces = local_domain_struct.phy_faces
    self.phy_faces_name = local_domain_struct.phy_faces_name
    self.phy_faces_loctoglob = local_domain_struct.phy_faces_loctoglob
    self.bf_cellid = local_domain_struct.bf_cellid
    self.cell_loctoglob = local_domain_struct.cell_loctoglob
    self.node_loctoglob = local_domain_struct.node_loctoglob
    self.halo_neighsub = local_domain_struct.halo_neighsub
    self.node_halos = local_domain_struct.node_halos
    self.node_halobfid = local_domain_struct.node_halobfid
    self.shared_bf_recv = local_domain_struct.shared_bf_recv
    self.bf_recv_part_size = local_domain_struct.bf_recv_part_size
    self.shared_bf_send = local_domain_struct.shared_bf_send
    self.halo_halosext = local_domain_struct.halo_halosext
    self.dim = local_domain_struct.dim
    self.float_precision = local_domain_struct.float_precision
    self.max_cell_nodeid = local_domain_struct.max_cell_nodeid
    self.max_cell_faceid = local_domain_struct.max_cell_faceid
    self.max_face_nodeid = local_domain_struct.max_face_nodeid
    self.max_node_haloid = local_domain_struct.max_node_haloid
    self.max_cell_halofid = local_domain_struct.max_cell_halofid
    self.max_cell_halonid = local_domain_struct.max_cell_halonid
    self.node_cellid = local_domain_struct.node_cellid
    self.cell_cellnid = local_domain_struct.cell_cellnid
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))

    self.start = time.time()

    print("bounds")
    self.bounds = self._define_bounds(self.nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("node_cellid")
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes, local_domain_struct.node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid, local_domain_struct.cell_cellnid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid
    ) = self._create_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    self.nb_faces = len(self.faces)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")


    print("Create cell volume and center")
    (
      self.cell_volume,
      self.cell_center
    ) = self._create_cell_info(self.cells, self.nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Face measure face center face normal")
    (
      self.face_measure,
      self.face_center,
      self.face_normal,
      self.face_tangent, # only in 3D, shape is 0 in 2D
      self.face_binormal # only in 3D, shape is 0 in 2D
    ) = self._create_face_info(self.faces, self.nodes, self.face_cellid, self.cell_center)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_halofid, cell_halonid, face_haloid, node_haloid")
    (
      self.cell_halofid,
      self.cell_halonid,
      self.face_haloid,
      self.node_haloid
    ) = self._create_halo_cells(self.cells, self.cell_faceid, self.faces, self.nodes, self.node_halos)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create face and node names")
    (
      self.face_oldname,
      self.node_oldname,
      self.face_name,
      self.node_name
    ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces, self.face_haloid, self.node_haloid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create shared_ghost_info")
    (
      self.shared_ghost_info,
      self.ghost_part_size
    ) = self._create_shared_ghost_info(self.bf_cellid, self.bf_recv_part_size, self.cell_center, self.cell_faceid, self.face_oldname, self.face_normal, self.face_center, self.face_measure, self.rank, len(self.shared_bf_recv))
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create shared_ghost_info")
    (
      self.node_ghostid,
      self.cell_ghostid,
      self.node_ghostcenter,
      self.face_ghostcenter,
      self.node_ghostfaceinfo
    ) = self._create_ghost_tables(self.shared_ghost_info, self.faces, self.cell_faceid, self.ghost_part_size)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")


    print("_share_ghost_info")
    self._share_ghost_info(self.rank, self.bf_recv_part_size, self.shared_ghost_info, self.shared_bf_send)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_halo_ghost_tables")
    (
      self.cell_haloghostnid,
      self.cell_haloghostcenter,
      self.node_haloghostid,
      self.node_haloghostcenter,
      self.node_haloghostfaceinfo
    ) = self._create_halo_ghost_tables(self.shared_ghost_info, self.cells, self.bf_cellid, self.node_halobfid, self.cell_faceid, self.faces, self.ghost_part_size)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    ## TODO the use of this tables !?
    self.node_periodicid = np.zeros((self.nb_nodes, 2), dtype=np.int32)
    self.cell_periodicnid = np.zeros((self.nb_cells, 2), dtype=np.int32)
    self.cell_periodicfid = np.zeros(self.nb_cells, dtype=np.int32)
    self.cell_shift = np.zeros((self.nb_cells, 3), dtype=self.float_precision)

    print("face_gradient_info")
    (
      self.face_air_diamond,
      self.face_param1,
      self.face_param2,
      self.face_param3,
      self.face_param4,
      self.face_f1,
      self.face_f2,
      self.face_f3,
      self.face_f4 # TODO 2D
    ) = self._face_gradient_info(self.face_cellid, self.faces, self.face_ghostcenter, self.face_name, self.face_normal, self.cell_center, self.halo_centvol, self.face_haloid, self.nodes, self.cell_shift)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_halo_ghost_tables")
    (
      self.node_R_x,
      self.node_R_y,
      self.node_R_z, # TODO 2D
      self.node_lambda_x,
      self.node_lambda_y,
      self.node_lambda_z, #TODO 2D
      self.node_number,
    ) = self._variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid, self.node_haloghostid, self.node_periodicid, self.nodes, self.face_ghostcenter, self.cell_haloghostcenter, self.halo_centvol, self.cell_shift)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_update_boundaries")
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
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_normal_face_of_cell_2d")
    # only in 2D, shape is 0 in 3D
    self.cell_nf = self._create_normal_face_of_cell_2d(self.cell_center, self.face_center, self.cell_faceid, self.face_normal)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_dist_ortho_function_2d")
    # only in 2D, shape is 0 in 3D
    self.face_dist_ortho = self._dist_ortho_function_2d(self.innerfaces, self.boundaryfaces, self.face_cellid, self.cell_center, self.face_center, self.face_normal)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")



  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int', g_node_cellid):
    if g_node_cellid is not None:
      return g_node_cellid
    return Domain.create_node_cellid(cells, nb_nodes)


  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]', g_cell_cellnid):
    if g_cell_cellnid is not None:
      return g_cell_cellnid
    return Domain.create_cell_cellnid(cells, node_cellid)

  @staticmethod
  def _create_info(
          cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int'
  ):
    # ? Create tables
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(nb_cells, max_cell_faceid, max_face_nodeid), dtype=np.int32)
    # tmp_size_info = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)
    apprx_nb_faces = nb_cells * max_cell_faceid # TODO ((nb_cells * max_cell_faceid + boundary_faces) / 2)
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    faces_counter = np.zeros(shape=1, dtype=np.int32)

    create_info(
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
    cell_center = np.zeros(shape=(nb_cells, self.dim), dtype=self.float_precision)
    if self.dim == 2:
      compute_cell_center_volume_2d(cells, nodes, cell_volume, cell_center)
    else:
      compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center)
    return (
      cell_volume,
      cell_center
    )

  def _create_face_info(self, faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]', cell_center: 'float[:]'):
    nb_faces = len(faces)
    face_measure = np.zeros(shape=nb_faces, dtype=self.float_precision)
    face_center = np.zeros(shape=(nb_faces, self.dim), dtype=self.float_precision)
    face_normal = np.zeros(shape=(nb_faces, self.dim), dtype=self.float_precision)
    face_tangent = np.zeros(shape=0, dtype=self.float_precision)
    face_binormal = np.zeros(shape=0, dtype=self.float_precision)
    if self.dim == 2:
      compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=self.float_precision)
      compute_face_info_3d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal, face_tangent, face_binormal)
    return (
      face_measure,
      face_center,
      face_normal,
      face_tangent,
      face_binormal
    )

  def _create_halo_cells(self, cells, cell_faceid, faces, nodes, node_halos):
    nb_cells = len(cells)
    nb_faces = len(faces)
    nb_nodes = len(nodes)

    cell_halofid = np.zeros(shape=(nb_cells, self.max_cell_halofid + 1), dtype=np.int32)
    cell_halonid = np.zeros(shape=(nb_cells, self.max_cell_halonid + 1), dtype=np.int32)
    face_haloid = np.zeros(shape=nb_faces, dtype=np.int32)
    node_haloid = np.zeros(shape=(nb_nodes, self.max_node_haloid + 1), dtype=np.int32)

    create_halo_cells(cells, cell_faceid, faces, node_halos, node_haloid, cell_halofid, cell_halonid, face_haloid)

    return (
      cell_halofid,
      cell_halonid,
      face_haloid,
      node_haloid
    )

  def _define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 face_haloid: 'int[:]',
                                 node_haloid: 'int[:, :]'
                                 ):
    nb_nodes = self.nb_nodes
    face_name = np.zeros(shape=faces.shape[0], dtype=np.int32)
    node_name = np.zeros(shape=nb_nodes, dtype=np.int32)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=np.int32)
    node_oldname = np.zeros(shape=nb_nodes, dtype=np.int32)

    # count max_node_faceid
    tmp = np.zeros(shape=nb_nodes, dtype=np.int32)
    get_max_node_faceid(phy_faces, tmp)
    max_node_faceid = np.max(tmp)

    # create node_phyfaceid
    node_phyfaceid = np.zeros(shape=(nb_nodes, max_node_faceid + 1), dtype=np.int32)
    get_node_faceid(phy_faces, node_phyfaceid)

    define_face_and_node_name(phy_faces, phy_faces_name, faces, node_phyfaceid, face_haloid, node_haloid, face_oldname, node_oldname, face_name, node_name)
    return (
      face_oldname,
      node_oldname,
      face_name,
      node_name
    )

  def _create_shared_ghost_info(self, bf_cellid: 'int[:, :]', bf_recv_part_size: 'int[:]', cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', rank: 'int', shared_bf_recv_size: 'int'):
    
    ghost_part_size = np.zeros(shape=2, dtype=np.int32)
    get_ghost_part_size(bf_recv_part_size, rank, ghost_part_size)

    if self.dim == 2:
      shared_ghost_info_data_size = 10
      shared_ghost_info = np.zeros(shape=(shared_bf_recv_size, shared_ghost_info_data_size), dtype=self.float_precision)

      create_ghost_info_2d(bf_cellid, cell_center, cell_faceid, face_oldname, face_normal, face_center, face_measure, shared_ghost_info, ghost_part_size[0])
    else:
      shared_ghost_info_data_size = 13
      shared_ghost_info = np.zeros(shape=(shared_bf_recv_size, shared_ghost_info_data_size), dtype=self.float_precision)

      create_ghost_info_3d(bf_cellid, cell_center, cell_faceid, face_oldname, face_normal, face_center, face_measure, shared_ghost_info, ghost_part_size[0])
    
    return (shared_ghost_info, ghost_part_size)

  def _create_ghost_tables(self, shared_ghost_info: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', ghost_part_size: 'int[:]'):

    start = ghost_part_size[0]
    end = start + ghost_part_size[1]

    node_nb_ghostid = np.zeros(shape=self.nb_nodes, dtype=np.int32)
    cell_nb_ghostid = np.zeros(shape=self.nb_cells, dtype=np.int32)
    get_ghost_tables_size(shared_ghost_info, faces, cell_faceid, node_nb_ghostid, cell_nb_ghostid, start, end)

    nb_faces = len(faces)

    max_node_ghost = np.max(node_nb_ghostid)
    max_cell_ghost = np.max(cell_nb_ghostid)

    node_ghostid = np.zeros(shape=(self.nb_nodes, max_node_ghost + 1), dtype=np.int32)
    cell_ghostid = np.zeros(shape=(self.nb_cells, max_cell_ghost + 1), dtype=np.int32)

    if self.dim == 2:
      node_ghostcenter_data_size = 5  # [ghost_center x.y, cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 3  # [ghost_center x.y, gamma]
      node_ghostfaceinfo_data_size = 4  # [face_center x.y, face_normal x.y]
      node_ghostcenter = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size), dtype=self.float_precision)
      face_ghostcenter = np.zeros(shape=(nb_faces, face_ghostcenter_data_size), dtype=self.float_precision)
      node_ghostfaceinfo = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size), dtype=self.float_precision)

      create_ghost_tables_2d(shared_ghost_info, faces, cell_faceid, node_ghostid, cell_ghostid, node_ghostcenter, face_ghostcenter, node_ghostfaceinfo, start, end)
    else:
      node_ghostcenter_data_size = 6  # [ghost_center x.y.z, cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 4  # [ghost_center x.y.z, gamma]
      node_ghostfaceinfo_data_size = 6  # [face_center x.y.z, face_normal x.y.z]
      node_ghostcenter = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size),
                                  dtype=self.float_precision)
      face_ghostcenter = np.zeros(shape=(nb_faces, face_ghostcenter_data_size), dtype=self.float_precision)
      node_ghostfaceinfo = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size),
                                    dtype=self.float_precision)

      create_ghost_tables_3d(shared_ghost_info, faces, cell_faceid, node_ghostid, cell_ghostid, node_ghostcenter, face_ghostcenter, node_ghostfaceinfo, start, end)

    return (
      node_ghostid,
      cell_ghostid,
      node_ghostcenter,
      face_ghostcenter,
      node_ghostfaceinfo
    )

  def _share_ghost_info(self, rank: 'int', bf_recv_part_size: 'int[:]', shared_ghost_info: 'float[:, :]', shared_bf_send: 'int[:]'):
    comm = MPI.COMM_WORLD

    shared_ghost_info_data_size = shared_ghost_info.shape[1] # 13 on 3D 10 on 2D
    recv_data = []
    reqs = []

    # ------------------------------------------------------------------
    # 1. Post non-blocking receives
    # ------------------------------------------------------------------
    start = 0
    for i in range(0, len(bf_recv_part_size), 2):
      the_sender = bf_recv_part_size[i]
      size = bf_recv_part_size[i + 1]
      if the_sender != rank:
        buffer = np.zeros(shape=(size, shared_ghost_info_data_size), dtype=self.float_precision)
        recv_data.append((start, buffer))
        req = comm.Irecv([buffer, MPI.FLOAT], source=the_sender, tag=0)
        reqs.append(req)
      start += size


    # ------------------------------------------------------------------
    # 2. Post non-blocking sends
    # ------------------------------------------------------------------
    i = 0
    while i < len(shared_bf_send):
      dest_part = shared_bf_send[i]
      start = i + 2
      end = start + shared_bf_send[i + 1]
      data_indices = shared_bf_send[start:end]
      data = shared_ghost_info[data_indices]
      req = comm.Isend([data, MPI.FLOAT], dest=dest_part, tag=0)
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
    # 4. Copy Data to shared_ghost_info
    # ------------------------------------------------------------------
    for item in recv_data:
      start = item[0]
      data = item[1]
      end = start + len(data)
      shared_ghost_info[start:end] = data

  def _create_halo_ghost_tables(self, shared_ghost_info: 'float[:, :]', cells: 'int[:, :]', bf_cellid: 'int[:, :]', node_halobfid: 'int[:, :]', cell_faceid, faces, ghost_part_size):
    nb_nodes = self.nb_nodes
    nb_cells = self.nb_cells
    shared_ghost_info_size = shared_ghost_info.shape[0]

    # ------------------------------------------------------------------
    # create_bcell_halobfid
    # ------------------------------------------------------------------
    visited = np.zeros(shape=shared_ghost_info_size, dtype=np.int32)
    max_bcell_halobfid = count_max_bcell_halobfid(cells, bf_cellid, node_halobfid, visited)

    bcell_halobfid = np.zeros(shape=(bf_cellid.shape[0], max_bcell_halobfid + 1), dtype=np.int32)
    visited.fill(0)
    create_bcell_halobfid(cells, bf_cellid, node_halobfid, visited, bcell_halobfid)

    # ------------------------------------------------------------------
    # create_bf_nodeid
    # ------------------------------------------------------------------
    visited = np.zeros(shape=nb_nodes, dtype=np.int8)
    max_bf_nodeid = count_max_bf_nodeid(bf_cellid, cell_faceid, faces, visited)

    bf_nodeid = np.zeros(shape=max_bf_nodeid, dtype=np.int32)
    visited.fill(0)
    create_bf_nodeid(bf_cellid, cell_faceid, faces, visited, bf_nodeid)

    # ------------------------------------------------------------------
    # ghost_new_index
    # ------------------------------------------------------------------
    ghost_new_index = np.zeros(shape=shared_ghost_info_size, dtype=np.int32)
    nb_haloghost = create_ghost_new_index(ghost_part_size, ghost_new_index)

    # ------------------------------------------------------------------
    # create_halo_ghost_tables
    # ------------------------------------------------------------------
    cell_haloghostnid = np.zeros(shape=(nb_cells, max_bcell_halobfid + 1), dtype=np.int32)

    if self.dim == 2:
      cell_haloghostcenter_data_size = 2
      node_haloghostcenter_data_size = 5
      node_haloghostfaceinfo_data_size = 4
      cell_haloghostcenter = np.zeros(shape=(nb_haloghost, cell_haloghostcenter_data_size), dtype=self.float_precision)
      node_haloghostid = np.zeros(shape=(nb_nodes, node_halobfid.shape[1]), dtype=np.int32)
      node_haloghostcenter = np.zeros(shape=(nb_nodes, node_halobfid.shape[1], node_haloghostcenter_data_size), dtype=self.float_precision)
      node_haloghostfaceinfo = np.zeros(shape=(nb_nodes, node_halobfid.shape[1], node_haloghostfaceinfo_data_size), dtype=self.float_precision)

      create_halo_ghost_tables_2d(shared_ghost_info, bcell_halobfid, bf_nodeid, node_halobfid, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter, node_haloghostfaceinfo)

    else:
      cell_haloghostcenter_data_size = 3
      node_haloghostcenter_data_size = 6
      node_haloghostfaceinfo_data_size = 6
      cell_haloghostcenter = np.zeros(shape=(nb_haloghost, cell_haloghostcenter_data_size), dtype=self.float_precision)
      node_haloghostid = np.zeros(shape=(nb_nodes, node_halobfid.shape[1]), dtype=np.int32)
      node_haloghostcenter = np.zeros(shape=(nb_nodes, node_halobfid.shape[1], node_haloghostcenter_data_size),
                                      dtype=self.float_precision)
      node_haloghostfaceinfo = np.zeros(shape=(nb_nodes, node_halobfid.shape[1], node_haloghostfaceinfo_data_size),
                                        dtype=self.float_precision)

      create_halo_ghost_tables_3d(shared_ghost_info, bcell_halobfid, bf_nodeid, node_halobfid, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter,node_haloghostfaceinfo)

    return (
      cell_haloghostnid,
      cell_haloghostcenter,
      node_haloghostid,
      node_haloghostcenter,
      node_haloghostfaceinfo
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
      face_gradient_info_2d(face_cellid, faces, face_ghostcenter, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_param4, face_f1, face_f2, face_f3, face_f4, cell_shift)
    else:
      face_gradient_info_3d(face_cellid, faces, face_ghostcenter, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_f1, face_f2, cell_shift)

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

  def _variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, face_ghostcenter, cell_haloghostcenter, halo_centvol, cell_shift):

    node_R_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_number = np.zeros(self.nb_nodes, dtype=np.int32)

    if self.dim == 2:
      variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

    return (
      node_R_x,
      node_R_y,
      node_R_z,
      node_lambda_x,
      node_lambda_y,
      node_lambda_z,
      node_number
    )

  def _create_normal_face_of_cell_2d(self, cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int32[:,:]', face_normal: 'float[:,:]'):

    cell_nf = np.zeros(shape=0, dtype=self.float_precision)
    if self.dim == 2:
      cell_nf = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 2), dtype=self.float_precision)
      create_normal_face_of_cell_2d(cell_center, face_center, cell_faceid, face_normal)
    return cell_nf

  def _dist_ortho_function_2d(self, d_innerfaces: 'uint32[:]', d_boundaryfaces: 'uint32[:]', face_cellid: 'int32[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]'):

    face_dist_ortho = np.zeros(shape=0, dtype=self.float_precision)
    if self.dim == 2:
      face_dist_ortho = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 2), dtype=self.float_precision)
      dist_ortho_function_2d(d_innerfaces, d_boundaryfaces, face_cellid, cell_center, face_center, face_normal, face_dist_ortho)
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



