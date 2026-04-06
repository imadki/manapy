import numpy as np
from numba.typed import Dict, List
from manapy.domain.MeshClass import Mesh
import manapy.domain.utils as utils
from manapy.domain.LocalDomainInterface import LocalDomainInterface
import warnings
import manapy.backends.types as types
import manapy.c_api.manapy_c_api as manapy_c_api

class Partitioning:

  Par_Graph_K_Way = 0
  Par_Dual = 1
  Par_Nodal = 2
  Par_Mgmetis = 3
  def __init__(self, mesh: 'Mesh'):
    self.nodes = mesh.points
    self.cells = mesh.cells
    self.cells_type = mesh.cells_type
    self.max_cell_nodeid = types.np_int_type(mesh.max_cell_nodeid)
    self.max_cell_faceid = types.np_int_type(mesh.max_cell_faceid)
    self.max_face_nodeid = types.np_int_type(mesh.max_face_nodeid)
    self.phy_faces = mesh.phy_faces
    self.phy_faces_name = mesh.phy_faces_name
    self.dim = mesh.dim
    self.nb_nodes = types.np_int_type(len(self.nodes))
    self.nb_cells = types.np_int_type(len(self.cells))
    self.nb_phy_faces = types.np_int_type(len(self.phy_faces))
    self.node_cellid = utils.create_node_cellid(self.cells, self.nb_nodes)
    self.node_phyid = utils.create_node_phyid(self.phy_faces, self.nb_nodes)

    (max_node_phyid, max_cell_phyid) = utils.get_max_phyid(len(self.cells), self.phy_faces, self.node_cellid, self.node_phyid)
    self.max_node_phyid = types.np_int_type(max_node_phyid)
    self.max_cell_phyid = types.np_int_type(max_cell_phyid)
    # Initialized when calling one of these functions (make_n_part_graph_k_way, make_n_part_mesh_dual, make_n_part_mesh_nodal)
    self.part_vert = None
    self.nb_parts = 1 # default value for one partition

  def _remap_part_vert(self, part_vert, nb_parts):
    # ####################################################################################
    # Remap the partition labels in `part_vert`
    # `part_vert` will be updated to hold the corresponding new indices in [0, len(unique_vals) - 1].
    unique_vals, part_vert = np.unique(part_vert, return_inverse=True)
    part_vert = part_vert.astype(types.np_int_type)
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
    cell_cellfid = utils.create_cellfid(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
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

  def set_part_vert(self, nb_parts, partitioning_type, n_common=None):
    if partitioning_type == self.Par_Mgmetis:
      return self.make_n_part_old(nb_parts)
    elif partitioning_type == self.Par_Dual:
      if n_common is None:
        raise TypeError("n_common must be specified")
      return self.make_n_part_mesh_dual(nb_parts, n_common)
    elif partitioning_type == self.Par_Nodal:
      return self.make_n_part_mesh_nodal(nb_parts)
    elif partitioning_type == self.Par_Graph_K_Way:
      return self.make_n_part_graph_k_way(nb_parts)
    raise ValueError(f"Unknown partitioning type: {partitioning_type}")

  def _create_one_partition(self):
    local_domains = LocalDomainInterface.new_local_domains(1)
    local_domain = local_domains[0]

    node_oldname = utils.define_node_oldname(self.phy_faces, self.phy_faces_name, self.nb_nodes)

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
    local_domain.halo_neighsub = np.zeros(shape=(1, 1), dtype=types.np_int_type)
    local_domain.node_halos = np.zeros(shape=1, dtype=types.np_int_type)
    # node_halophyid
    local_domain.halo_halosext = np.zeros(shape=(1, 1), dtype=types.np_int_type)
    local_domain.halo_halosint = np.zeros(shape=1, dtype=types.np_int_type)
    local_domain.halo_centvol = np.zeros(shape=(1, 1), dtype=types.np_float_type)
    #local_domain.cell_loctoglob = np.zeros(shape=0, dtype=types.np_int_type) # keep it shape=0
    local_domain.cell_loctoglob = np.arange(0, self.nb_cells, dtype=types.np_int_type)
    local_domain.cell_tc = np.arange(0, self.nb_cells, dtype=types.np_int_type)
    local_domain.node_loctoglob = np.zeros(shape=1, dtype=types.np_int_type)

    local_domain.phyid_recv = np.arange(self.nb_phy_faces, dtype=types.np_int_type)
    local_domain.phyid_recv_part_size = np.array([0, self.nb_phy_faces], dtype=types.np_int_type)
    local_domain.node_halophyid = np.zeros(shape=(1, 1), dtype=types.np_int_type)
    #local_domain.phyid_send = np.zeros(shape=1, dtype=types.np_int_type)


    local_domain.max_node_haloid = 0 # NONE
    local_domain.max_cell_halonid = 0 # NONE
    local_domain.max_cell_halophyid = 0 # NONE
    local_domain.max_node_halophyid = 0 # NONE



    return local_domains


  def create_sub_domains(self):
    if self.nb_parts == 1:
      return self._create_one_partition()
    else: # multiple partitions
      if self.part_vert is None:
        raise RuntimeError("must call one of these functions make_n_part_graph_k_way, make_n_part_mesh_dual, make_n_part_mesh_nodal")
      return LocalDomainInterface.create_local_domains_wrapper(
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

