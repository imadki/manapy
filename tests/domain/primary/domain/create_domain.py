import numpy as np
import meshio
import time
import h5py
import os
import shutil
import manapy_domain
from mpi4py import MPI
from manapy.ddm.geometry   import Face, Cell, Node, Halo
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
                                 get_cell_ghostnid_size,
                                 create_cell_ghostnid,
                                 get_ghost_part_size,
                                 get_ghost_tables_size,
                                 count_max_bcell_halobfid,
                                 create_bcell_halobfid,
                                 count_max_b_nodeid,
                                 create_b_nodeid,
                                 create_ghost_new_index,
                                 create_halo_ghost_tables_2d,
                                 create_halo_ghost_tables_3d,
                                 face_gradient_info_2d,
                                 face_gradient_info_3d,
                                 variables_2d,
                                 variables_3d,
                                 create_normal_face_of_cell_2d,
                                 dist_ortho_function_2d,
                                 get_max_b_ncellid,
                                 create_b_ncellid
                                 )


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
    self.halo_halosext = None # [[global index of halocell, global index of cell nodes, size]] shape=(nb_halos, max_cell_nodeid + 2)
    self.halo_centvol = None # [halocell_center_{x, y, z}, halocell_volume_{x, y, z}] # z axis only on 3D
    self.dim = None
    self.float_precision = None
    self.max_cell_nodeid = None
    self.max_cell_faceid = None
    self.max_face_nodeid = None
    self.max_node_haloid = None
    self.max_cell_halofid = None
    self.max_cell_halonid = None




  def save_hdf5(self, path):
    with h5py.File(path, 'w') as f:
      f.create_dataset('nodes', data=self.nodes)
      f.create_dataset('cells', data=self.cells)
      f.create_dataset('cells_type', data=self.cells_type)
      f.create_dataset('phy_faces', data=self.phy_faces)
      f.create_dataset('phy_faces_name', data=self.phy_faces_name)
      f.create_dataset('phy_faces_loctoglob', data=self.phy_faces_loctoglob)
      f.create_dataset('bf_cellid', data=self.bf_cellid)
      f.create_dataset('cell_loctoglob', data=self.cell_loctoglob)
      f.create_dataset('node_loctoglob', data=self.node_loctoglob)
      f.create_dataset('halo_neighsub', data=self.halo_neighsub)
      f.create_dataset('node_halos', data=self.node_halos)
      f.create_dataset('node_halobfid', data=self.node_halobfid)
      f.create_dataset('shared_bf_recv', data=self.shared_bf_recv)
      f.create_dataset('bf_recv_part_size', data=self.bf_recv_part_size)
      f.create_dataset('shared_bf_send', data=self.shared_bf_send)
      f.create_dataset('halo_halosext', data=self.halo_halosext)
      f.create_dataset('halo_centvol', data=self.halo_centvol)
      f.create_dataset('dim', data=self.dim)
      f.create_dataset('float_precision', data=self.float_precision)
      f.create_dataset('max_cell_nodeid', data=self.max_cell_nodeid)
      f.create_dataset('max_cell_faceid', data=self.max_cell_faceid)
      f.create_dataset('max_face_nodeid', data=self.max_face_nodeid)
      f.create_dataset('max_node_haloid', data=self.max_node_haloid)
      f.create_dataset('max_cell_halofid', data=self.max_cell_halofid)
      f.create_dataset('max_cell_halonid', data=self.max_cell_halonid)

  @staticmethod
  def load_hd5(path: 'str'):
    local_domain = LocalDomainStruct()

    with h5py.File(path, 'r') as f:
      local_domain.nodes = f['nodes'][...]
      local_domain.cells = f['cells'][...]
      local_domain.cells_type = f['cells_type'][...]
      local_domain.phy_faces = f['phy_faces'][...]
      local_domain.phy_faces_name = f['phy_faces_name'][...]
      local_domain.phy_faces_loctoglob = f['phy_faces_loctoglob'][...]
      local_domain.bf_cellid = f['bf_cellid'][...]
      local_domain.cell_loctoglob = f['cell_loctoglob'][...]
      local_domain.node_loctoglob = f['node_loctoglob'][...]
      local_domain.halo_neighsub = f['halo_neighsub'][...]
      local_domain.node_halos = f['node_halos'][...]
      local_domain.node_halobfid = f['node_halobfid'][...]
      local_domain.shared_bf_recv = f['shared_bf_recv'][...]
      local_domain.bf_recv_part_size = f['bf_recv_part_size'][...]
      local_domain.shared_bf_send = f['shared_bf_send'][...]
      local_domain.halo_halosext = f['halo_halosext'][...]
      local_domain.halo_centvol = f['halo_centvol'][...]
      local_domain.dim = f['dim'][()]
      local_domain.float_precision = f['float_precision'][()].decode()
      local_domain.max_cell_nodeid = f['max_cell_nodeid'][()]
      local_domain.max_cell_faceid = f['max_cell_faceid'][()]
      local_domain.max_face_nodeid = f['max_face_nodeid'][()]
      local_domain.max_node_haloid = f['max_node_haloid'][()]
      local_domain.max_cell_halofid = f['max_cell_halofid'][()]
      local_domain.max_cell_halonid = f['max_cell_halonid'][()]

    return local_domain

class GlobalDomain:

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

    self.start = time.time() # For timing only


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
    return GlobalDomain.create_node_cellid(bf_nodes, nb_nodes)



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

  def _create_one_partition(self):
    local_domain = LocalDomainStruct()

    local_domain.nodes = self.nodes
    local_domain.cells = self.cells
    local_domain.cells_type = self.cells_type
    local_domain.phy_faces = self.phy_faces
    local_domain.phy_faces_name = self.phy_faces_name
    local_domain.dim = self.dim
    local_domain.float_precision = self.float_precision
    local_domain.max_cell_nodeid = self.max_cell_nodeid
    local_domain.max_cell_faceid = self.max_cell_faceid
    local_domain.max_face_nodeid = self.max_face_nodeid

    ## Halo related tables
    local_domain.phy_faces_loctoglob = np.zeros(shape=0, dtype=np.int32)
    local_domain.halo_neighsub = np.zeros(shape=0, dtype=np.int32)
    local_domain.node_halos = np.zeros(shape=(1, 1), dtype=np.int32)
    local_domain.cell_loctoglob = np.zeros(shape=0, dtype=np.int32)
    local_domain.node_loctoglob = np.zeros(shape=0, dtype=np.int32)
    local_domain.bf_cellid = np.zeros(shape=(1, 1), dtype=np.int32) # It will be created on LocalDomain Class
    local_domain.shared_bf_recv = np.arange(self.nb_phy_faces, dtype=np.int32)
    local_domain.bf_recv_part_size = np.array([0, self.nb_phy_faces], dtype=np.int32)
    local_domain.node_halobfid = np.zeros(shape=(1, 1), dtype=np.int32)
    local_domain.shared_bf_send = np.zeros(shape=0, dtype=np.int32)
    local_domain.halo_halosext = np.zeros(shape=(1, 1), dtype=np.int32)
    local_domain.halo_centvol = np.zeros(shape=(1, 1), dtype=np.float32)
    local_domain.max_node_haloid = 0 # NONE
    local_domain.max_cell_halofid = 0 # NONE
    local_domain.max_cell_halonid = 0 # NONE



    return local_domain

  def _create_multiple_partitions(self, nb_parts: 'int'):
    print("node_cellid")
    node_cellid = self.create_node_cellid(self.cells, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    cell_cellnid = self.create_cell_cellnid(self.cells, node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellfid and boundary_faces")
    (
      cell_cellfid,
      bf_cellid,
      bf_nodes
    ) = self._create_cellfid_bf_info(self.cells, node_cellid, self.cells_type, self.max_cell_faceid,
                                     self.max_face_nodeid, self.nb_phy_faces)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("node_bfid")  # node_boundary_face_id
    node_bfid = self.create_node_bfid(bf_nodes, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Start creating sub domains")
    res = manapy_domain.create_sub_domains(
      cell_cellfid,
      node_cellid,
      node_bfid,
      bf_cellid,
      self.cells,
      cell_cellfid,
      cell_cellnid,
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
      l_domain.halo_centvol = self._create_halocentvol(l_domain.halo_halosext, self.nodes)
      l_domain.dim = self.dim
      l_domain.float_precision = self.float_precision
      l_domain.max_cell_nodeid = item[16]
      l_domain.max_cell_faceid = item[17]
      l_domain.max_face_nodeid = item[18]
      l_domain.max_node_haloid = item[19]
      l_domain.max_cell_halofid = item[20]
      l_domain.max_cell_halonid = item[21]
      local_domains.append(l_domain)

    print(f"Execution time: {time.time() - self.start:.6f} seconds")
    return local_domains

  def c_create_sub_domains(self, nb_parts: 'int'):
    if nb_parts == 1:
      return [self._create_one_partition()]
    else:
      return self._create_multiple_partitions(nb_parts)




  def _create_halocentvol(self, halo_halosext, nodes):
    halo_cells = halo_halosext[:, 1:] # exclude cellid and keep nodeids and size

    nb_halo_cells = len(halo_cells)
    cell_volume = np.zeros(shape=nb_halo_cells, dtype=self.float_precision)
    cell_center = np.zeros(shape=(nb_halo_cells, self.dim), dtype=self.float_precision)
    halo_centvol = np.zeros(shape=(nb_halo_cells, 4), dtype=self.float_precision)
    if self.dim == 2:
      compute_cell_center_volume_2d(halo_cells, nodes, cell_volume, cell_center)
      halo_centvol[:, 0:2] = cell_center
      halo_centvol[:, 3] = cell_volume
    else:
      compute_cell_center_volume_3d(halo_cells, nodes, cell_volume, cell_center)
      halo_centvol[:, 0:3] = cell_center
      halo_centvol[:, 3] = cell_volume


    return halo_centvol

  def create_and_save_local_domains(self, size: 'int'):
    local_domains_data = self.c_create_sub_domains(size)  # Number of partitions

    folder_name = f"local_domain_{size}"
    if not os.path.exists(folder_name):
      os.makedirs(folder_name, exist_ok=True)
    for rank in range(size):
      file_name = f"mesh{rank}.hdf5"
      path = os.path.join(folder_name, file_name)
      local_domains_data[rank].save_hdf5(path)

  @staticmethod
  def all_local_mesh_files_exist(size: int):
    folder_name = f"local_domain_{size}"
    for rank in range(size):
      file_path = os.path.join(folder_name, f"mesh{rank}.hdf5")
      if not os.path.isfile(file_path):
        return False
    return True

  @staticmethod
  def delete_local_domain_folder(size: int):
    folder_name = f"local_domain_{size}"
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
      shutil.rmtree(folder_name)

# TODO: face_haloid, node_haloid rename
class LocalDomain:
  def __init__(self, local_domain_struct: 'LocalDomainStruct', rank: 'int', size: 'int'):
    self.rank = rank
    self.size = size
    self.nodes = local_domain_struct.nodes
    self.cells = local_domain_struct.cells
    self.cells_type = local_domain_struct.cells_type
    self.phy_faces = local_domain_struct.phy_faces
    self.phy_faces_name = local_domain_struct.phy_faces_name
    self.bf_cellid = local_domain_struct.bf_cellid # reassign In case size = 1
    self.cell_loctoglob = local_domain_struct.cell_loctoglob
    self.node_loctoglob = local_domain_struct.node_loctoglob
    self.halo_neighsub = local_domain_struct.halo_neighsub
    self.node_halos = local_domain_struct.node_halos
    self.node_halobfid = local_domain_struct.node_halobfid
    self.shared_bf_recv = local_domain_struct.shared_bf_recv
    self.bf_recv_part_size = local_domain_struct.bf_recv_part_size
    self.shared_bf_send = local_domain_struct.shared_bf_send
    self.halo_halosext = local_domain_struct.halo_halosext
    self.halo_centvol = local_domain_struct.halo_centvol
    self.dim = local_domain_struct.dim
    self.float_precision = local_domain_struct.float_precision
    self.max_cell_nodeid = local_domain_struct.max_cell_nodeid
    self.max_cell_faceid = local_domain_struct.max_cell_faceid
    self.max_face_nodeid = local_domain_struct.max_face_nodeid
    self.max_node_haloid = local_domain_struct.max_node_haloid
    self.max_cell_halofid = local_domain_struct.max_cell_halofid
    self.max_cell_halonid = local_domain_struct.max_cell_halonid
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))
    self.nb_phy_faces = np.int32(len(self.phy_faces))

    self.start = time.time()

    print("bounds")
    self.bounds = self._define_bounds(self.nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("node_cellid")
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid,
      self.bf_cellid
    ) = self._create_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid, self.bf_cellid)
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
    ) = self._create_shared_ghost_info(self.bf_cellid, self.bf_recv_part_size, self.cell_center, self.cell_faceid, self.cell_loctoglob, self.face_oldname, self.face_normal, self.face_center, self.face_measure, self.rank, len(self.shared_bf_recv))
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create shared_ghost_info")
    (
      self.node_ghostid,
      self.cell_ghostnid,
      self.node_ghostcenter,
      self.face_ghostcenter,
      self.node_ghostfaceinfo
    ) = self._create_ghost_tables(self.shared_ghost_info, self.cells, self.faces, self.cell_faceid, self.ghost_part_size)
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
    ) = self._create_halo_ghost_tables(self.shared_ghost_info, self.cells, self.phy_faces, self.node_cellid, self.node_halobfid, self.node_haloid, self.halo_halosext,self.ghost_part_size)
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
    ) = self._variables(self.cell_center, self.node_cellid, self.node_haloid, self.node_ghostid, self.node_haloghostid, self.node_periodicid, self.nodes, self.node_oldname, self.face_ghostcenter, self.cell_haloghostcenter, self.halo_centvol, self.cell_shift)
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

    print("_define_BCs")
    self.BCs = self._define_BCs(self.periodicinfaces, self.periodicupperfaces, self.periodicfrontfaces)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_normal_face_of_cell_2d")
    # only in 2D, shape is 0 in 3D
    self.cell_nf = self._create_normal_face_of_cell_2d(self.cell_center, self.face_center, self.cell_faceid, self.face_normal)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_dist_ortho_function_2d")
    # only in 2D, shape is 0 in 3D
    self.face_dist_ortho = self._dist_ortho_function_2d(self.innerfaces, self.boundaryfaces, self.face_cellid, self.cell_center, self.face_center, self.face_normal)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")



  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int'):
    return GlobalDomain.create_node_cellid(cells, nb_nodes)


  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    return GlobalDomain.create_cell_cellnid(cells, node_cellid)


  def _create_info(self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int',
    origin_bf_cellid: 'int[:, :]',
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
    bf_cellid = np.zeros(shape=(self.nb_phy_faces, 2), dtype=np.int8) # int8
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
      faces_counter,
      bf_cellid,
      self.size
    )

    faces = faces[:faces_counter[0]]
    face_cellid = face_cellid[:faces_counter[0]]

    if self.size != 1:
      bf_cellid = origin_bf_cellid

    return (
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid,
      bf_cellid
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

    if self.size == 1:
      cell_halofid = np.zeros(shape=0, dtype=np.int32)
      cell_halonid = np.zeros(shape=0, dtype=np.int32)
      face_haloid = np.zeros(shape=0, dtype=np.int32)
      node_haloid = np.zeros(shape=(nb_nodes, 1), dtype=np.int32)
    else:
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

  def _create_shared_ghost_info(self, bf_cellid: 'int[:, :]', bf_recv_part_size: 'int[:]', cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]', face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]', face_measure: 'float[:]', rank: 'int', shared_bf_recv_size: 'int'):
    ghost_part_size = np.zeros(shape=2, dtype=np.int32)
    get_ghost_part_size(bf_recv_part_size, rank, ghost_part_size)

    if self.dim == 2:
      shared_ghost_info_data_size = 11
      shared_ghost_info = np.zeros(shape=(shared_bf_recv_size, shared_ghost_info_data_size), dtype=self.float_precision)

      create_ghost_info_2d(bf_cellid, cell_center, cell_faceid, cell_loctoglob, face_oldname, face_normal, face_center, face_measure, shared_ghost_info, ghost_part_size[0])
    else:
      shared_ghost_info_data_size = 14
      shared_ghost_info = np.zeros(shape=(shared_bf_recv_size, shared_ghost_info_data_size), dtype=self.float_precision)

      create_ghost_info_3d(bf_cellid, cell_center, cell_faceid, cell_loctoglob, face_oldname, face_normal, face_center, face_measure, shared_ghost_info, ghost_part_size[0])

    return (shared_ghost_info, ghost_part_size)

  def _create_ghost_tables(self, shared_ghost_info: 'int[:, :]', cells: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', ghost_part_size: 'int[:]'):

    start = ghost_part_size[0]
    end = start + ghost_part_size[1]

    node_nb_ghostid = np.zeros(shape=self.nb_nodes, dtype=np.int32)
    get_ghost_tables_size(shared_ghost_info, faces, cell_faceid, node_nb_ghostid, start, end)

    max_node_ghost = np.max(node_nb_ghostid)
    node_ghostid = np.zeros(shape=(self.nb_nodes, max_node_ghost + 1), dtype=np.int32)

    # ------------------------------------------------------------------
    #  node_ghostid
    #  node_ghostcenter
    #  face_ghostcenter
    #  node_ghostfaceinfo
    # ------------------------------------------------------------------

    if self.dim == 2:
      node_ghostcenter_data_size = 5  # [ghost_center x.y, cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 3  # [ghost_center x.y, gamma]
      node_ghostfaceinfo_data_size = 4  # [face_center x.y, face_normal x.y]
      node_ghostcenter = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size), dtype=self.float_precision)
      face_ghostcenter = np.zeros(shape=(self.nb_faces, face_ghostcenter_data_size), dtype=self.float_precision)
      node_ghostfaceinfo = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size), dtype=self.float_precision)

      create_ghost_tables_2d(shared_ghost_info, faces, cell_faceid, node_ghostid, node_ghostcenter, face_ghostcenter, node_ghostfaceinfo, start, end)
    else:
      node_ghostcenter_data_size = 6  # [ghost_center x.y.z, cell_id, face_old_name, face_id]
      face_ghostcenter_data_size = 4  # [ghost_center x.y.z, gamma]
      node_ghostfaceinfo_data_size = 6  # [face_center x.y.z, face_normal x.y.z]
      node_ghostcenter = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostcenter_data_size),
                                  dtype=self.float_precision)
      face_ghostcenter = np.zeros(shape=(self.nb_faces, face_ghostcenter_data_size), dtype=self.float_precision)
      node_ghostfaceinfo = np.zeros(shape=(self.nb_nodes, max_node_ghost, node_ghostfaceinfo_data_size),
                                    dtype=self.float_precision)

      create_ghost_tables_3d(shared_ghost_info, faces, cell_faceid, node_ghostid, node_ghostcenter, face_ghostcenter, node_ghostfaceinfo, start, end)

    # ------------------------------------------------------------------
    # cell_ghostnid
    # ------------------------------------------------------------------

    bc_visited = np.zeros(shape=self.nb_cells, dtype=np.int8)
    ghost_visited = np.ones(shape=self.nb_faces, dtype=np.int32) * -1
    cell_ghostnid_size = np.zeros(shape=self.nb_cells, dtype=np.int32)
    get_cell_ghostnid_size(cells, node_ghostid, bc_visited, ghost_visited, cell_ghostnid_size)

    bc_visited.fill(0)
    ghost_visited.fill(-1)
    max_cell_ghostnid = np.max(cell_ghostnid_size)
    cell_ghostnid = np.zeros(shape=(self.nb_cells, max_cell_ghostnid + 1), dtype=np.int32)

    create_cell_ghostnid(cells, node_ghostid, bc_visited, ghost_visited, cell_ghostnid)


    return (
      node_ghostid,
      cell_ghostnid,
      node_ghostcenter,
      face_ghostcenter,
      node_ghostfaceinfo
    )

  def _share_ghost_info(self, rank: 'int', bf_recv_part_size: 'int[:]', shared_ghost_info: 'float[:, :]', shared_bf_send: 'int[:]'):
    if self.size == 1:
      return

    comm = MPI.COMM_WORLD

    shared_ghost_info_data_size = shared_ghost_info.shape[1] # 14 on 3D 11 on 2D
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

  def _create_halo_ghost_tables(self, shared_ghost_info: 'float[:, :]', cells: 'int[:, :]', phy_faces: 'int[:, :]', node_cellid: 'int[:, :]', node_halobfid: 'int[:, :]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', ghost_part_size):
    nb_nodes = self.nb_nodes
    nb_cells = self.nb_cells

    if self.size == 1:
      cell_haloghostnid = np.zeros(shape=(1, 1), dtype=np.int32)
      cell_haloghostcenter = np.zeros(shape=(1, 1), dtype=self.float_precision)
      node_haloghostid = np.zeros(shape=(nb_nodes, 1), dtype=np.int32)
      node_haloghostcenter = np.zeros(shape=(1, 1, 1), dtype=self.float_precision)
      node_haloghostfaceinfo = np.zeros(shape=(1, 1, 1), dtype=self.float_precision)

      self.bcell_halobfid = np.zeros(shape=(1, 1), dtype=np.int32) # TODO remove
    else:
      shared_ghost_info_size = shared_ghost_info.shape[0]

      # ------------------------------------------------------------------
      # create_b_nodeid
      # ------------------------------------------------------------------
      visited = np.zeros(shape=nb_nodes, dtype=np.int8)
      max_b_nodeid = count_max_b_nodeid(phy_faces, visited)
      visited.fill(0)
      b_nodeid = np.zeros(shape=max_b_nodeid, dtype=np.int32)
      create_b_nodeid(phy_faces, visited, b_nodeid)


      # ------------------------------------------------------------------
      # create_b_ncellid
      # ------------------------------------------------------------------
      visited = np.zeros(shape=len(cells), dtype=np.int8)
      max_b_ncellid = get_max_b_ncellid(b_nodeid, node_cellid, visited)
      visited.fill(0)
      b_ncellid = np.zeros(shape=max_b_ncellid, dtype=np.int32)
      create_b_ncellid(b_nodeid, node_cellid, visited, b_ncellid)

      # TODO check all visited for -1
      visited = np.ones(shape=shared_ghost_info_size, dtype=np.int32) * -1
      max_bcell_halobfid = count_max_bcell_halobfid(cells, b_ncellid, node_halobfid, visited)

      bcell_halobfid = np.zeros(shape=(b_ncellid.shape[0], max_bcell_halobfid + 2), dtype=np.int32)
      visited.fill(-1)
      create_bcell_halobfid(cells, b_ncellid, node_halobfid, visited, bcell_halobfid)
      self.bcell_halobfid = bcell_halobfid # TODO remove

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

        create_halo_ghost_tables_2d(shared_ghost_info, bcell_halobfid, b_nodeid, node_halobfid, node_haloid, halo_halosext, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter, node_haloghostfaceinfo)

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

        create_halo_ghost_tables_3d(shared_ghost_info, bcell_halobfid, b_nodeid, node_halobfid, node_haloid, halo_halosext, ghost_new_index, cell_haloghostnid, cell_haloghostcenter, node_haloghostid, node_haloghostcenter,node_haloghostfaceinfo)
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

  def _variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, cell_shift):

    node_R_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_R_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_x = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_y = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_lambda_z = np.zeros(self.nb_nodes, dtype=self.float_precision)
    node_number = np.zeros(self.nb_nodes, dtype=np.int32)

    if self.dim == 2:
      variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, face_ghostcenter, cell_haloghostcenter, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

    return (
      node_R_x,
      node_R_y,
      node_R_z,
      node_lambda_x,
      node_lambda_y,
      node_lambda_z,
      node_number
    )

  def _create_normal_face_of_cell_2d(self, cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int[:,:]', face_normal: 'float[:,:]'):

    cell_nf = np.zeros(shape=0, dtype=self.float_precision)
    if self.dim == 2:
      cell_nf = np.zeros(shape=(self.nb_cells, self.max_cell_faceid, 2), dtype=self.float_precision)
      create_normal_face_of_cell_2d(cell_center, face_center, cell_faceid, face_normal, cell_nf)
    return cell_nf

  def _dist_ortho_function_2d(self, d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]'):

    face_dist_ortho = np.zeros(shape=0, dtype=self.float_precision)
    if self.dim == 2:
      face_dist_ortho = np.zeros(shape=self.nb_faces, dtype=self.float_precision)
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
    "d_node_halobfid", # TODO remove
    "d_bcell_halobfid", # TODO remove
    # "d_cell_tc",
    "d_node_loctoglob",
    "d_node_cellid",
    "d_node_name",
    "d_node_oldname",
    "d_node_ghostid",
    "d_node_haloghostid",
    "d_node_ghostcenter",
    "d_node_haloghostcenter",
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
    "float_precision"
  ]

  def __init__(self, nb_partitions, mesh_name, float_precision, dim, create_par_fun):
    if create_par_fun:
      create_par_fun(nb_partitions, mesh_name, float_precision=float_precision, dim=dim)

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

class Domain:
  def __init__(self, local_domain: 'LocalDomain'):

    self.node_halobfid = local_domain.node_halobfid # TODO remove
    self.bcell_halobfid = local_domain.bcell_halobfid # TODO remove

    # Init
    self.float_precision = local_domain.float_precision
    self.rank = local_domain.rank
    self.size = local_domain.size
    self.dim = local_domain.dim
    self.nbnodes = local_domain.nb_nodes
    self.nbcells = local_domain.nb_cells
    self.nbfaces = local_domain.nb_faces
    self.nbhalos = local_domain.halo_halosext.shape[0]
    self._maxcellfid = local_domain.max_cell_faceid
    self._maxcellnodeid = local_domain.max_cell_nodeid
    self._maxfacenid = local_domain.max_face_nodeid

    # TODO Latter
    self.backend = None
    self.conf = None
    self.int_precision = None
    self.mpi_precision = None
    self.comm = None
    self.forcedbackend = None
    self.signature = None
    self.vtkprecision = None
    self._parameters = None
    self._vtkpath = None

    self.cells = Cell()
    self.nodes = Node()
    self.faces = Face()
    self.halos = Halo()

    # Cells
    self.cells._nbcells = local_domain.nb_cells
    self.cells._nodeid = local_domain.cells
    self.cells._faceid = local_domain.cell_faceid
    self.cells._cellfid = local_domain.cell_cellfid
    self.cells._cellnid = local_domain.cell_cellnid
    self.cells._halonid = local_domain.cell_halonid
    self.cells._ghostnid = local_domain.cell_ghostnid
    self.cells._haloghostnid = local_domain.cell_haloghostnid
    self.cells._haloghostcenter = local_domain.cell_haloghostcenter
    self.cells._center = local_domain.cell_center
    self.cells._volume = local_domain.cell_volume
    self.cells._nf = local_domain.cell_nf
    self.cells._globtoloc = None
    self.cells._loctoglob = local_domain.cell_loctoglob
    self.cells._tc = None
    self.cells._periodicfid = local_domain.cell_periodicfid
    self.cells._shift = local_domain.cell_shift

    # Nodes
    self.nodes._nbnodes = local_domain.nb_nodes
    self.nodes._vertex = local_domain.nodes
    self.nodes._name = local_domain.node_name
    self.nodes._oldname = local_domain.node_oldname
    self.nodes._cellid = local_domain.node_cellid
    self.nodes._ghostid = local_domain.node_ghostid
    self.nodes._haloghostid = local_domain.node_haloghostid
    self.nodes._ghostcenter = local_domain.node_ghostcenter
    self.nodes._haloghostcenter = local_domain.node_haloghostcenter
    self.nodes._ghostfaceinfo = local_domain.node_ghostfaceinfo
    self.nodes._haloghostfaceinfo = local_domain.node_haloghostfaceinfo
    self.nodes._loctoglob = local_domain.node_loctoglob
    self.nodes._halonid = local_domain.node_haloid
    self.nodes._nparts = None
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
    self.faces._normal = local_domain.face_normal
    self.faces._mesure = local_domain.face_measure
    self.faces._center = local_domain.face_center
    self.faces._dist_ortho = local_domain.face_dist_ortho
    self.faces._ghostcenter = local_domain.face_ghostcenter
    self.faces._oppnodeid = None
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

    # Halos
    self.halos._halosint = np.zeros(shape=10) # TODO
    self.halos._halosext = local_domain.halo_halosext
    self.halos._neigh = local_domain.halo_neighsub
    self.halos._centvol = local_domain.halo_centvol
    self.halos._sizehaloghost = local_domain.cell_haloghostcenter.shape[0]
    # TODO
    self.halos._scount = None
    self.halos._rcount = None
    self.halos._indsend = None
    self.halos._comm_ptr = None
    self.halos._faces = None
    self.halos._nodes = None
    self.halos._requests = None

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
    self.typeOfCells = None
    self.bounds = self._bounds

  @staticmethod
  def partitioning(mesh_path, dim, float_precision, size: 'int'):
    GlobalDomain.delete_local_domain_folder(size)
    mesh = Mesh(mesh_path, dim)
    domain = GlobalDomain(mesh, float_precision)
    domain.c_create_sub_domains(size)
    # domain.create_and_save_local_domains(size)

  @staticmethod
  def create_domain(mesh_path, dim, float_precision, recreate=True):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
      # try:
      mesh = Mesh(mesh_path, dim)
      domain = GlobalDomain(mesh, float_precision)
      local_domain_data = domain.c_create_sub_domains(1)
      local_domain = LocalDomain(local_domain_data[0], rank, size)
      return Domain(local_domain)
      # except Exception as e:
      #   print(f"Failed: {e}")
      #   exit(1)
    else:
      status = 0

      if rank == 0:
        print("====> Start <=====")
        try:
          if not (recreate == False and GlobalDomain.all_local_mesh_files_exist(size)):
            print("====> Creating Mesh <=====")
            GlobalDomain.delete_local_domain_folder(size)
            mesh = Mesh(mesh_path, dim)
            domain = GlobalDomain(mesh, float_precision)
            domain.create_and_save_local_domains(size)
            print("====> End <=====")
        except Exception as e:
          print(f"[Rank 0] failed: {e}")
          status = 1

      # Broadcast rank 0's status to all
      status = comm.bcast(status, root=0)

      # Now all ranks wait and check status
      if status != 0:
        comm.Abort(1)


      local_domain = LocalDomain.load_and_create(rank, size)

      # TODO ckeck if local_domain failed and abort
      return Domain(local_domain)
