import os
from manapy.domain.compute import *
from numba.typed import Dict, List
import manapy.c_api.manapy_c_api as manapy_c_api
from manapy.backends.types import FLOAT_TYPE
from manapy.domain.MeshClass import Mesh
import manapy.domain.utils as utils
from manapy.domain.LocalDomainStructClass import LocalDomainStruct
import warnings

class PartitioningUtils:

  def __init__(self, mesh: 'Mesh'):
    self.nodes = mesh.points
    self.cells = mesh.cells
    self.cells_type = mesh.cells_type
    self.max_cell_nodeid = np.int32(mesh.max_cell_nodeid)
    self.max_cell_faceid = np.int32(mesh.max_cell_faceid)
    self.max_face_nodeid = np.int32(mesh.max_face_nodeid)
    self.phy_faces = mesh.phy_faces
    self.phy_faces_name = mesh.phy_faces_name
    self.dim = mesh.dim
    self.float_precision = FLOAT_TYPE
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))
    self.nb_phy_faces = np.int32(len(self.phy_faces))
    self.node_cellid = utils.create_node_cellid(self.cells, self.nb_nodes)
    self.node_phyid = utils.create_node_phyid(self.phy_faces, self.nb_nodes)
    # Initialized when calling one of these functions (make_n_part_graph_k_way, make_n_part_mesh_dual, make_n_part_mesh_nodal)
    self.part_vert = None
    self.nb_parts = 1 # default value for one partition


  # ###############################
  # ###############################

  @staticmethod
  def _create_cellfid(
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int32',
    max_face_nodeid: 'int32'
  ):
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    # tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)

    create_cellfid(
      cells,
      node_cellid,
      cell_type,
      max_cell_faceid,
      max_face_nodeid,
      cell_cellfid
    )

    return cell_cellfid


  def _define_node_oldname(self, phy_faces, phy_faces_name):
    node_oldname = np.zeros(shape=self.nb_nodes, dtype=np.int32)
    define_node_oldname(phy_faces, phy_faces_name, node_oldname)

    return node_oldname

  @staticmethod
  def _create_local_domains_wrapper(part_vert: 'int32[:]', node_cellid: 'int32[:, :]', node_phyid: 'int32[:, :]', cells: 'int32[:, :]', cells_type: 'int8[:]', nodes: 'float[:, :]', phy_faces: 'int32[:, :]', phy_faces_name: 'int32[:]', nb_parts: 'int32', float_precision: 'int32', dim: 'int32'):
    c_res = manapy_c_api.create_local_domains(part_vert, node_cellid, node_phyid, cells, cells_type, nodes, phy_faces, phy_faces_name, nb_parts, dim)

    list_local_domains = LocalDomainStruct.new_local_domains(nb_parts)
    for i in range(nb_parts):
      obj = list_local_domains[i]

      k = 0
      obj.nodes = c_res[i][k]; k+=1
      obj.cells = c_res[i][k]; k+=1
      obj.cells_type = c_res[i][k]; k+=1
      obj.phy_faces = c_res[i][k]; k+=1
      obj.phy_faces_name = c_res[i][k]; k+=1
      obj.cell_loctoglob = c_res[i][k]; k+=1
      obj.node_loctoglob = c_res[i][k]; k+=1
      obj.node_oldname = c_res[i][k]; k+=1
      obj.halo_neighsub = c_res[i][k]; k+=1
      obj.node_halos = c_res[i][k]; k+=1
      obj.node_halophyid = c_res[i][k]; k+=1
      obj.phyid_recv = c_res[i][k]; k+=1
      obj.phyid_recv_part_size = c_res[i][k]; k+=1
      obj.phyid_send = c_res[i][k]; k+=1
      obj.halo_halosext = c_res[i][k]; k+=1
      obj.halo_halosint = c_res[i][k]; k+=1
      obj.halo_centvol = c_res[i][k]; k+=1
      obj.max_cell_nodeid = c_res[i][k]; k+=1
      obj.max_cell_faceid = c_res[i][k]; k+=1
      obj.max_face_nodeid = c_res[i][k]; k+=1
      obj.max_node_haloid = c_res[i][k]; k+=1
      obj.max_cell_halonid = c_res[i][k]; k+=1
      obj.dim = dim
      obj.float_precision = float_precision

    return list_local_domains

  def _remap_part_vert(self, part_vert, nb_parts):
    # ####################################################################################
    # Remap the partition labels in `part_vert`
    # `part_vert` will be updated to hold the corresponding new indices in [0, len(unique_vals) - 1].
    unique_vals, part_vert = np.unique(part_vert, return_inverse=True)
    part_vert = part_vert.astype(np.int32)
    if len(unique_vals) != nb_parts:
      warnings.warn(
        f"The original number of partitions (nb_parts={nb_parts}) was changed by METIS to {len(unique_vals)}. "
        f"This means some partitions have no cells. Forcing the number of partitions to {len(unique_vals)}."
      )
    nb_parts = len(unique_vals)
    self.part_vert = part_vert
    self.nb_parts = nb_parts

class Partitioning(PartitioningUtils):
  def __init__(self, mesh: 'Mesh'):
    super().__init__(mesh)

  def make_n_part_graph_k_way(self, nb_parts):
    if nb_parts <= 1:
      return
    cell_cellfid = self._create_cellfid(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    part_vert = manapy_c_api.make_n_part_graph_k_way(cell_cellfid, nb_parts)
    return self._remap_part_vert(part_vert, nb_parts)

  def make_n_part_mesh_dual(self, nb_parts, n_common):
    if nb_parts <= 1:
      return
    part_vert = manapy_c_api.make_n_part_mesh_dual(self.cells, nb_parts, n_common)
    return self._remap_part_vert(part_vert, nb_parts)

  def make_n_part_mesh_nodal(self, nb_parts):
    if nb_parts <= 1:
      return
    part_vert = manapy_c_api.make_n_part_mesh_nodal(self.cells, nb_parts)
    return self._remap_part_vert(part_vert, nb_parts)


  def make_n_part_old(self, nb_parts):
    if nb_parts <= 1:
      return
    from mgmetis import metis
    from mgmetis.enums import OPTION

    # Partitioning mesh
    opts = metis.get_default_options()
    opts[OPTION.MINCONN] = 1
    opts[OPTION.CONTIG] = 1
    opts[OPTION.NUMBERING] = 0
    opts[OPTION.OBJTYPE] = 1
    options = opts


    elem = [sublist[0:sublist[-1]] for sublist in self.cells]
    _, e_part, _ = metis.part_mesh_dual(nb_parts, elem, opts=options, nv=self.nb_nodes)
    return self._remap_part_vert(e_part, nb_parts)

  def _create_one_partition(self):
    local_domains = LocalDomainStruct.new_local_domains(1)
    local_domain = local_domains[0]

    node_oldname = self._define_node_oldname(self.phy_faces, self.phy_faces_name)

    local_domain.nodes = self.nodes
    local_domain.cells = self.cells
    local_domain.cells_type = self.cells_type
    local_domain.phy_faces = self.phy_faces
    local_domain.phy_faces_name = self.phy_faces_name
    local_domain.node_oldname = node_oldname
    local_domain.max_cell_nodeid = self.max_cell_nodeid
    local_domain.max_cell_faceid = self.max_cell_faceid
    local_domain.max_face_nodeid = self.max_face_nodeid
    local_domain.dim = self.dim
    local_domain.float_precision = 32 if self.float_precision == 'float32' else 64

    ## Halo related tables
    local_domain.halo_neighsub = np.zeros(shape=(1, 1), dtype=np.int32)
    local_domain.node_halos = np.zeros(shape=1, dtype=np.int32)
    # node_halophyid
    local_domain.halo_halosext = np.zeros(shape=(1, 1), dtype=np.int32)
    local_domain.halo_halosint = np.zeros(shape=1, dtype=np.int32)
    local_domain.halo_centvol = np.zeros(shape=(1, 1), dtype=np.float64)
    local_domain.cell_loctoglob = np.zeros(shape=0, dtype=np.int32) # keep it shape=0
    local_domain.node_loctoglob = np.zeros(shape=1, dtype=np.int32)

    local_domain.phyid_recv = np.arange(self.nb_phy_faces, dtype=np.int32)
    local_domain.phyid_recv_part_size = np.array([0, self.nb_phy_faces], dtype=np.int32)
    local_domain.node_halophyid = np.zeros(shape=(1, 1), dtype=np.int32)
    #local_domain.phyid_send = np.zeros(shape=1, dtype=np.int32)

    local_domain.max_node_haloid = 0 # NONE
    local_domain.max_cell_halonid = 0 # NONE



    return local_domains

  def create_sub_domains(self):
    if self.nb_parts == 1:
      return self._create_one_partition()
    else: # multiple partitions
      if self.part_vert is None:
        raise RuntimeError("must call one of these functions make_n_part_graph_k_way, make_n_part_mesh_dual, make_n_part_mesh_nodal")
      return self._create_local_domains_wrapper(
        self.part_vert,
        self.node_cellid,
        self.node_phyid,
        self.cells,
        self.cells_type,
        self.nodes,
        self.phy_faces,
        self.phy_faces_name,
        self.nb_parts,
        32 if self.float_precision == 'float32' else 64,
        self.dim
      )

  @staticmethod
  def save_local_domains(local_domains, nb_parts: 'int'):
    folder_name = f"local_domain_{nb_parts}"
    if not os.path.exists(folder_name):
      os.makedirs(folder_name, exist_ok=True)
    for rank in range(nb_parts):
      file_name = f"mesh{rank}.hdf5"
      path = os.path.join(folder_name, file_name)
      LocalDomainStruct.save_hdf5(local_domains[rank], path)
