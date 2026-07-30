import numpy as np
from manapy.domain.MeshClass import Mesh
from manapy.domain.LocalDomainInterface import LocalDomainInterface
import warnings
from manapy.compute import DomainCompute

class Partitioning:
  Par_Graph_K_Way = 0
  Par_Dual = 1 # n_common=3 by default
  Par_Nodal = 2
  def __init__(self, mesh: 'Mesh'):
    self.config = mesh.config # Same config as mesh
    self.domain_compute = DomainCompute(self.config)

    self.nodes = mesh.points
    self.cells = mesh.cells
    self.cells_type = mesh.cells_type
    self.max_cell_nodeid = self.config.int_dtype(mesh.max_cell_nodeid)
    self.max_cell_faceid = self.config.int_dtype(mesh.max_cell_faceid)
    self.max_face_nodeid = self.config.int_dtype(mesh.max_face_nodeid)
    self.phy_faces = mesh.phy_faces
    self.phy_faces_name = mesh.phy_faces_name
    self.dim = mesh.dim
    self.nb_nodes = self.config.int_dtype(len(self.nodes))
    self.nb_cells = self.config.int_dtype(len(self.cells))
    self.nb_phy_faces = self.config.int_dtype(len(self.phy_faces))
    self.node_cellid = self.domain_compute.create_node_cellid(self.cells, self.nb_nodes)
    self.node_phyid = self.domain_compute.create_node_phyid(self.phy_faces, self.nb_nodes)

    (max_node_phyid, max_cell_phyid) = self.domain_compute.get_max_phyid(len(self.cells), self.phy_faces, self.node_cellid, self.node_phyid)
    self.max_node_phyid = self.config.int_dtype(max_node_phyid)
    self.max_cell_phyid = self.config.int_dtype(max_cell_phyid)
    # Initialized when calling one of these functions (make_n_part_graph_k_way, make_n_part_mesh_dual, make_n_part_mesh_nodal)
    self.part_vert = None
    self.nb_parts = 1 # default value for one partition

  def _remap_part_vert(self, part_vert, nb_parts):
    # ####################################################################################
    # Remap the partition labels in `part_vert`
    # `part_vert` will be updated to hold the corresponding new indices in [0, len(unique_vals) - 1].
    unique_vals, part_vert = np.unique(part_vert, return_inverse=True)
    part_vert = part_vert.astype(self.config.int_dtype)
    if len(unique_vals) != nb_parts:
      warnings.warn(
        f"The original number of partitions (nb_parts={nb_parts}) was changed by METIS to {len(unique_vals)}. "
        f"This means some partitions have no cells. Forcing the number of partitions to {len(unique_vals)}."
      )
      raise ValueError(f"The original number of partitions (nb_parts={nb_parts}) was changed please choose a smaller one.")
    nb_parts = len(unique_vals)
    self.part_vert = part_vert
    self.nb_parts = nb_parts
    return part_vert, nb_parts

  def make_n_part_graph_k_way(self, nb_parts):
    if nb_parts <= 1:
      return
    cell_cellfid = self.domain_compute.create_cellfid(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    part_vert = self.domain_compute.make_n_part_graph_k_way(cell_cellfid, nb_parts)
    return self._remap_part_vert(part_vert, nb_parts)

  def make_n_part_mesh_dual(self, nb_parts, n_common):
    if nb_parts <= 1:
      return
    part_vert = self.domain_compute.make_n_part_mesh_dual(self.cells, nb_parts, n_common)
    return self._remap_part_vert(part_vert, nb_parts)

  def make_n_part_mesh_nodal(self, nb_parts):
    if nb_parts <= 1:
      return
    part_vert = self.domain_compute.make_n_part_mesh_nodal(self.cells, nb_parts)
    return self._remap_part_vert(part_vert, nb_parts)

  def set_part_vert(self, nb_parts, partitioning_type, n_common=3):
    if partitioning_type == self.Par_Dual:
      if n_common is None:
        raise TypeError("n_common must be specified")
      return self.make_n_part_mesh_dual(nb_parts, n_common)
    elif partitioning_type == self.Par_Nodal:
      return self.make_n_part_mesh_nodal(nb_parts)
    elif partitioning_type == self.Par_Graph_K_Way:
      return self.make_n_part_graph_k_way(nb_parts)
    raise ValueError(f"Unknown partitioning type: {partitioning_type}")

  def _create_one_partition(self):
    local_domains = LocalDomainInterface.new_local_domains(1, self.config)
    local_domain = local_domains[0]

    node_oldname = self.domain_compute.define_node_oldname(self.phy_faces, self.phy_faces_name, self.nb_nodes)

    local_domain.nodes = self.nodes
    local_domain.cells = self.cells
    local_domain.cells_type = self.cells_type
    local_domain.phy_faces = self.phy_faces
    local_domain.phy_faces_name = self.phy_faces_name
    local_domain.node_oldname = node_oldname
    local_domain.max_cell_nodeid = self.max_cell_nodeid
    local_domain.max_cell_faceid = self.max_cell_faceid
    local_domain.max_face_nodeid = self.max_face_nodeid
    local_domain.max_cell_phyid = self.max_cell_phyid
    local_domain.max_node_phyid = self.max_node_phyid
    local_domain.dim = self.dim

    ## Halo related tables
    local_domain.halo_neighsub = np.zeros(shape=(0, 1), dtype=self.config.int_dtype)
    local_domain.node_halos = np.zeros(shape=0, dtype=self.config.int_dtype)
    # node_halophyid
    local_domain.halo_halosext = np.zeros(shape=(0, 1), dtype=self.config.int_dtype)
    local_domain.halo_halosint = np.zeros(shape=0, dtype=self.config.int_dtype)
    local_domain.halo_centvol = np.zeros(shape=(0, 4), dtype=self.config.float_dtype)
    #local_domain.cell_loctoglob = np.zeros(shape=0, dtype=self.config.int_dtype) # keep it shape=0
    local_domain.cell_loctoglob = np.arange(0, self.nb_cells, dtype=self.config.int_dtype)
    local_domain.cell_tc = np.arange(0, self.nb_cells, dtype=self.config.int_dtype)
    local_domain.node_loctoglob = np.zeros(shape=0, dtype=self.config.int_dtype)

    local_domain.phyid_recv = np.arange(self.nb_phy_faces, dtype=self.config.int_dtype)
    local_domain.phyid_recv_part_size = np.array([0, self.nb_phy_faces], dtype=self.config.int_dtype)
    local_domain.node_halophyid = np.zeros(shape=(0, 1), dtype=self.config.int_dtype)
    #local_domain.phyid_send = np.zeros(shape=1, dtype=types.np_int_type)


    local_domain.max_node_haloid = 0 # NONE
    local_domain.max_cell_halonid = 0 # NONE
    local_domain.max_cell_halophyid = 0 # NONE
    local_domain.max_node_halophyid = 0 # NONE



    return local_domains


  def _create_local_domains_wrapper(self, part_vert: 'int[:]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]', cells: 'int[:, :]', cells_type: 'int8[:]', nodes: 'float[:, :]', phy_faces: 'int[:, :]', phy_faces_name: 'int[:]', nb_parts: 'int', dim: 'int'):
    c_res = self.domain_compute.create_local_domains(part_vert, node_cellid, node_phyid, cells, cells_type, nodes, phy_faces, phy_faces_name, nb_parts, dim)

    list_local_domains = LocalDomainInterface.new_local_domains(nb_parts, self.config)
    counter, cell_tc = 0, np.zeros(len(part_vert), dtype=self.config.int_dtype)
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
      obj.halo_halosext = c_res[i][k]; k+=1
      obj.halo_halosint = c_res[i][k]; k+=1
      obj.halo_centvol = c_res[i][k]; k+=1
      obj.phyid_neighbor = c_res[i][k]; k+=1
      obj.phyid_recv = c_res[i][k]; k+=1
      obj.phyid_send = c_res[i][k]; k+=1
      obj.node_halophyid = c_res[i][k]; k+=1
      obj.cell_halophyid = c_res[i][k]; k+=1
      obj.max_cell_nodeid = c_res[i][k]; k+=1
      obj.max_cell_faceid = c_res[i][k]; k+=1
      obj.max_face_nodeid = c_res[i][k]; k+=1
      obj.max_node_haloid = c_res[i][k]; k+=1
      obj.max_cell_halonid = c_res[i][k]; k+=1
      obj.max_node_phyid = c_res[i][k]; k+=1
      obj.max_node_halophyid = c_res[i][k]; k+=1
      obj.max_cell_phyid = c_res[i][k]; k+=1
      obj.max_cell_halophyid = c_res[i][k]; k+=1
      obj.dim = dim

      # All ranks have cell_tc = array([], types.np_int_type)
      # Build tc for rank0
      cell_tc[counter:counter + len(obj.cells)] = obj.cell_loctoglob[:]
      counter += len(obj.cells)

    # Like the old version, only rank 0 stores cell_tc
    list_local_domains[0].cell_tc = cell_tc


    return list_local_domains

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
        self.dim
      )

