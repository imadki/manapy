from functools import partial

from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
import numpy as np
from typing import Callable
import manapy_compute_64_64
import manapy_compute_64_32
import manapy_compute_32_32
import manapy_compute_32_64

_manapy_compute = {
  "float64": {
    "int32": manapy_compute_64_32,
    "int64": manapy_compute_64_64,
  },
  "float32": {
    "int32": manapy_compute_32_32,
    "int64": manapy_compute_32_64,
  }
}


class _Compute:
  # TODO need description
  def __init__(self, config: ManapyConfig):
    self.config = config
    self.manapy_compute = _manapy_compute[self.config.float_precision][self.config.int_precision]

    # Functions
    self.compute_face_info_2d = self.manapy_compute.domain.compute_face_info_2d
    self.compute_face_info_3d = self.manapy_compute.domain.compute_face_info_3d
    self.count_max_bcell_halophyid = self.manapy_compute.domain.count_max_bcell_halophyid
    self.count_max_cell_cellnid = self.manapy_compute.domain.count_max_cell_cellnid
    self.count_max_node_cellid = self.manapy_compute.domain.count_max_node_cellid
    self.create_b_ncellid = self.manapy_compute.domain.create_b_ncellid
    self.create_bcell_halophyid = self.manapy_compute.domain.create_bcell_halophyid
    self.create_bf_cellid = self.manapy_compute.domain.create_bf_cellid
    self.create_cell_cellnid = self.manapy_compute.domain.create_cell_cellnid
    self.create_cellfid = self.manapy_compute.domain.create_cellfid
    self.create_ghost_info = self.manapy_compute.domain.create_ghost_info
    self.create_ghost_tables = self.manapy_compute.domain.create_ghost_tables
    self.create_halo_cells = self.manapy_compute.domain.create_halo_cells
    self.create_halo_ghost_tables = self.manapy_compute.domain.create_halo_ghost_tables
    self.create_info = self.manapy_compute.domain.create_info
    self.create_node_cellid = self.manapy_compute.domain.create_node_cellid
    self.create_normal_face_of_cell = self.manapy_compute.domain.create_normal_face_of_cell
    self.define_face_name = self.manapy_compute.domain.define_face_name
    self.define_node_oldname = self.manapy_compute.domain.define_node_oldname
    self.dist_ortho_function_2d = self.manapy_compute.domain.dist_ortho_function_2d
    self.face_gradient_info_2d = self.manapy_compute.domain.face_gradient_info_2d
    self.face_gradient_info_3d = self.manapy_compute.domain.face_gradient_info_3d
    self.fv_face_geometry = self.manapy_compute.domain.fv_face_geometry
    self.get_cell_nb_phyid = self.manapy_compute.domain.get_cell_nb_phyid
    self.get_max_b_ncellid = self.manapy_compute.domain.get_max_b_ncellid
    self.variables_2d = self.manapy_compute.domain.variables_2d
    self.variables_3d = self.manapy_compute.domain.variables_3d
    self.accum_periodic_dir = self.manapy_compute.domain.accum_periodic_dir
    self.node_periodic_bits = self.manapy_compute.domain.node_periodic_bits
    self.pair_periodic_faces = self.manapy_compute.domain.pair_periodic_faces
    self.compute_cell_center_area_2d = self.manapy_compute.domain.compute_cell_center_area_2d
    self.compute_cell_center_volume_3d = self.manapy_compute.domain.compute_cell_center_volume_3d

    self.make_n_part_graph_k_way = self.manapy_compute.partitioning.make_n_part_graph_k_way
    self.make_n_part_mesh_dual = self.manapy_compute.partitioning.make_n_part_mesh_dual
    self.make_n_part_mesh_nodal = self.manapy_compute.partitioning.make_n_part_mesh_nodal
    self.create_local_domains = self.manapy_compute.partitioning.create_local_domains


    # Variable Cpu
    self.facetocell = self.manapy_compute.core.facetocell
    self.celltoface = self.manapy_compute.core.celltoface
    self.barthlimiter_2d = self.manapy_compute.core.barthlimiter_2d
    self.cell_gradient_2d = self.manapy_compute.core.cell_gradient_2d
    self.center_to_vertex_2d = self.manapy_compute.core.center_to_vertex_2d
    self.face_gradient_2d = self.manapy_compute.core.face_gradient_2d
    self.vanalbadalimiter_2d = self.manapy_compute.core.vanalbadalimiter_2d
    self.barthlimiter_3d = self.manapy_compute.core.barthlimiter_3d
    self.cell_gradient_3d = self.manapy_compute.core.cell_gradient_3d
    self.center_to_vertex_3d = self.manapy_compute.core.center_to_vertex_3d
    self.face_gradient_3d = self.manapy_compute.core.face_gradient_3d
    self.vanalbadalimiter_3d = self.manapy_compute.core.vanalbadalimiter_3d
    # Variable Gpu
    self.facetocell_cuda = self.manapy_compute.core.facetocell_cuda
    self.celltoface_cuda = self.manapy_compute.core.celltoface_cuda
    self.barthlimiter_2d_cuda = self.manapy_compute.core.barthlimiter_2d_cuda
    self.cell_gradient_2d_cuda = self.manapy_compute.core.cell_gradient_2d_cuda
    self.center_to_vertex_2d_cuda = self.manapy_compute.core.center_to_vertex_2d_cuda
    self.face_gradient_2d_cuda = self.manapy_compute.core.face_gradient_2d_cuda
    self.vanalbadalimiter_2d_cuda = self.manapy_compute.core.vanalbadalimiter_2d_cuda
    self.barthlimiter_3d_cuda = self.manapy_compute.core.barthlimiter_3d_cuda
    self.cell_gradient_3d_cuda = self.manapy_compute.core.cell_gradient_3d_cuda
    self.center_to_vertex_3d_cuda = self.manapy_compute.core.center_to_vertex_3d_cuda
    self.face_gradient_3d_cuda = self.manapy_compute.core.face_gradient_3d_cuda
    self.vanalbadalimiter_3d_cuda = self.manapy_compute.core.vanalbadalimiter_3d_cuda

    # Boundary Cpu
    self.ghost_value_dirichlet = self.manapy_compute.boundary.ghost_value_dirichlet
    self.ghost_value_neumann = self.manapy_compute.boundary.ghost_value_neumann
    self.ghost_value_neumannNH = self.manapy_compute.boundary.ghost_value_neumannNH
    self.ghost_value_nonslip = self.manapy_compute.boundary.ghost_value_nonslip
    self.haloghost_value_dirichlet = self.manapy_compute.boundary.haloghost_value_dirichlet
    self.haloghost_value_neumann = self.manapy_compute.boundary.haloghost_value_neumann
    self.haloghost_value_neumannNH = self.manapy_compute.boundary.haloghost_value_neumannNH
    self.haloghost_value_nonslip = self.manapy_compute.boundary.haloghost_value_nonslip
    self.ghost_value_slip_2d = self.manapy_compute.boundary.ghost_value_slip_2d
    self.ghost_value_slip_3d = self.manapy_compute.boundary.ghost_value_slip_3d
    self.haloghost_value_slip_2d = self.manapy_compute.boundary.haloghost_value_slip_2d
    self.haloghost_value_slip_3d = self.manapy_compute.boundary.haloghost_value_slip_3d
    # Boundary Gpu
    self.ghost_value_dirichlet_cuda = self.manapy_compute.boundary.ghost_value_dirichlet_cuda
    self.ghost_value_neumann_cuda = self.manapy_compute.boundary.ghost_value_neumann_cuda
    self.ghost_value_neumannNH_cuda = self.manapy_compute.boundary.ghost_value_neumannNH_cuda
    self.ghost_value_nonslip_cuda = self.manapy_compute.boundary.ghost_value_nonslip_cuda
    self.haloghost_value_dirichlet_cuda = self.manapy_compute.boundary.haloghost_value_dirichlet_cuda
    self.haloghost_value_neumann_cuda = self.manapy_compute.boundary.haloghost_value_neumann_cuda
    self.haloghost_value_neumannNH_cuda = self.manapy_compute.boundary.haloghost_value_neumannNH_cuda
    self.haloghost_value_nonslip_cuda = self.manapy_compute.boundary.haloghost_value_nonslip_cuda
    self.ghost_value_slip_2d_cuda = self.manapy_compute.boundary.ghost_value_slip_2d_cuda
    self.ghost_value_slip_3d_cuda = self.manapy_compute.boundary.ghost_value_slip_3d_cuda
    self.haloghost_value_slip_2d_cuda = self.manapy_compute.boundary.haloghost_value_slip_2d_cuda
    self.haloghost_value_slip_3d_cuda = self.manapy_compute.boundary.haloghost_value_slip_3d_cuda

    ##############################################
    ######## Advection
    # Advection Cpu
    self.advec_explicitscheme_convective_2d = self.manapy_compute.solvers.advec.explicitscheme_convective_2d
    self.advec_explicitscheme_convective_3d = self.manapy_compute.solvers.advec.explicitscheme_convective_3d
    self.advec_time_step = self.manapy_compute.solvers.advec.time_step
    # Advection Gpu
    self.advec_explicitscheme_convective_2d_cuda = self.manapy_compute.solvers.advec.explicitscheme_convective_2d_cuda
    self.advec_explicitscheme_convective_3d_cuda = self.manapy_compute.solvers.advec.explicitscheme_convective_3d_cuda
    self.advec_time_step_cuda = self.manapy_compute.solvers.advec.time_step_cuda

    ##############################################
    ######## Solver utilities (common to every solver)
    # Solver utils Cpu
    # The two Gaussian-init kernels have no _cuda counterpart: they are CPU-only
    # (see src/solvers/headers/utils/utils_compute.hpp), so there is nothing to
    # list under "Gpu" below for them.
    self.initialisation_gaussian_2d = self.manapy_compute.solvers.utils.initialisation_gaussian_2d
    self.initialisation_gaussian_3d = self.manapy_compute.solvers.utils.initialisation_gaussian_3d
    self.update_new_value = self.manapy_compute.solvers.utils.update_new_value
    # Solver utils Gpu
    self.update_new_value_cuda = self.manapy_compute.solvers.utils.update_new_value_cuda


class DomainCompute:
  def __init__(self, config: ManapyConfig):
    self.config = config
    self.compute = _Compute(config)

    self.make_n_part_graph_k_way = self.compute.make_n_part_graph_k_way
    self.make_n_part_mesh_dual = self.compute.make_n_part_mesh_dual
    self.make_n_part_mesh_nodal = self.compute.make_n_part_mesh_nodal
    self.create_local_domains = self.compute.create_local_domains
    self.compute_cell_center_area_2d = self.compute.compute_cell_center_area_2d
    self.compute_cell_center_volume_3d = self.compute.compute_cell_center_volume_3d

  def create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int'):
    # Count max node cellid
    res = np.zeros(shape=nb_nodes, dtype=self.config.int_dtype)
    self.compute.count_max_node_cellid(cells, res)
    max_node_cellid = np.max(res)

    # Create node cellid
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=self.config.int_dtype)
    self.compute.create_node_cellid(cells, node_cellid)
    return node_cellid


  # LocalDomainClass.py
  def create_node_phyid(self, phy_faces: 'int[:, :]', nb_nodes: 'int'):
    # Count max node boundary faces
    # Create node boundary faceid
    return self.create_node_cellid(phy_faces, nb_nodes)

  # LocalDomainClass.py
  def create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    # Count max cell cellnid
    i_visited = np.ones(cells.shape[0], dtype=self.config.int_dtype) * -1
    max_cell_cellnid = self.compute.count_max_cell_cellnid(cells, node_cellid, i_visited)

    # Create cell cellnid
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=self.config.int_dtype)
    self.compute.create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  # Partitioning.py
  def get_max_phyid(self, nb_cells: 'int', phy_faces: 'int[:, :]', node_cellid: 'int[:, :]', node_phyid: 'int[:, :]'):
    i_visited = np.ones(shape=nb_cells, dtype=self.config.int_dtype) * -1
    cell_nb_phyid = np.zeros(shape=nb_cells, dtype=self.config.int_dtype)

    self.compute.get_cell_nb_phyid(phy_faces, node_cellid, i_visited, cell_nb_phyid)
    node_max_phyid = np.max(node_phyid[:, -1])
    cell_max_phyid = np.max(cell_nb_phyid)
    return node_max_phyid, cell_max_phyid

  # Partitioning.py
  def define_node_oldname(self, phy_faces, phy_faces_name, nb_nodes):
    node_oldname = np.zeros(shape=nb_nodes, dtype=self.config.int_dtype)
    self.compute.define_node_oldname(phy_faces, phy_faces_name, node_oldname)
    return node_oldname

  # Partitioning.py
  def create_cellfid(
    self,
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int'
  ):
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=self.config.int_dtype)
    # tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=self.config.int_dtype)
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)

    self.compute.create_cellfid(
      cells,
      node_cellid,
      cell_type,
      cell_cellfid
    )

    return cell_cellfid

  ############################################################################
  ############################################################################
  ############################################################################

  def create_info(self,
                   cells: 'int[:, :]',
                   node_cellid: 'int[:, :]',
                   cell_type: 'int[:]',
                   max_cell_faceid: 'int',
                   max_face_nodeid: 'int'
                   ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=self.config.int_dtype)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=self.config.int_dtype)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=self.config.int_dtype)
    apprx_nb_faces = nb_cells * max_cell_faceid
    faces = np.zeros(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=self.config.int_dtype)
    cell_faceid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=self.config.int_dtype) * -1
    cell_cellfid = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=self.config.int_dtype)
    faces_counter = np.zeros(shape=1, dtype=self.config.int_dtype)

    self.compute.create_info(
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

  def create_cell_info(self, cells, nodes, dim):
    nb_cells = len(cells)
    cell_volume = np.zeros(shape=nb_cells, dtype=self.config.float_dtype)
    cell_center = np.zeros(shape=(nb_cells, 3), dtype=self.config.float_dtype)
    if dim == 2:
      self.compute.compute_cell_center_area_2d(cells, nodes, cell_volume, cell_center)
    else:
      self.compute.compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center)
    return (
      cell_volume,
      cell_center
    )

  def create_face_info(self, faces: 'int[:, :]', nodes: 'float[:, :]', face_cellid: 'int[:, :]',
                        cell_center: 'float[:]', dim):
    nb_faces = len(faces)
    face_measure = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_center = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
    face_normal = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
    face_tangent = np.zeros(shape=0, dtype=self.config.float_dtype)
    face_binormal = np.zeros(shape=0, dtype=self.config.float_dtype)

    if dim == 2:
      self.compute.compute_face_info_2d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal)
    else:
      face_tangent = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
      face_binormal = np.zeros(shape=(nb_faces, 3), dtype=self.config.float_dtype)
      self.compute.compute_face_info_3d(faces, nodes, face_cellid, cell_center, face_measure, face_center, face_normal,
                                   face_tangent, face_binormal)
    return (
      face_measure,
      face_center,
      face_normal,
      face_tangent,
      face_binormal
    )

  def create_halo_cells(self, cells, faces, nodes, node_halos, halo_halosext, size, max_cell_halonid, max_node_haloid):
    nb_cells = len(cells)
    nb_faces = len(faces)
    nb_nodes = len(nodes)
    nb_halos = len(halo_halosext)

    if size == 1:
      # give size to cell_halonid, face_haloid and node_haloid to keep the multiprocessing code as it is
      cell_halonid = np.zeros(shape=(nb_cells, 1), dtype=self.config.int_dtype)
      face_haloid = np.ones(shape=nb_faces, dtype=self.config.int_dtype) * -1
      node_haloid = np.zeros(shape=(nb_nodes, 1), dtype=self.config.int_dtype)
    else:
      cell_halonid = np.zeros(shape=(nb_cells, max_cell_halonid + 1), dtype=self.config.int_dtype)
      face_haloid = np.zeros(shape=nb_faces, dtype=self.config.int_dtype)
      node_haloid = np.zeros(shape=(nb_nodes, max_node_haloid + 1), dtype=self.config.int_dtype)
      b_visited = np.zeros(shape=nb_halos, dtype=np.int8)

      self.compute.create_halo_cells(cells, faces, node_halos, node_haloid, b_visited, cell_halonid, face_haloid)

    return (
      cell_halonid,
      face_haloid,
      node_haloid
    )

  def define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 face_haloid: 'int[:]',
                                 node_haloid: 'int[:, :]',
                                 node_oldname: 'int[:]',
                                 nb_nodes
                                 ):
    face_name = np.zeros(shape=faces.shape[0], dtype=self.config.int_dtype)
    face_oldname = np.zeros(shape=faces.shape[0], dtype=self.config.int_dtype)
    phyid_to_faceid = np.ones(shape=phy_faces.shape[0], dtype=self.config.int_dtype) * -1
    face_to_phyid = np.ones(shape=faces.shape[0], dtype=self.config.int_dtype) * -1

    node_name = node_oldname.copy()
    if node_haloid.shape[0] != 0:
      node_name[node_haloid[:, -1] != 0] = 10

    node_phyid = self.create_node_phyid(phy_faces, nb_nodes)

    self.compute.define_face_name(phy_faces, phy_faces_name, faces, node_phyid, face_haloid, face_oldname, face_name, phyid_to_faceid, face_to_phyid)

    return (
      face_oldname,
      face_name,
      node_name,
      phyid_to_faceid,
      face_to_phyid
    )

  def create_ghost_info(self, cell_center: 'float[:, :]', cell_faceid: 'int[:, :]', cell_loctoglob: 'int[:]',
                         face_oldname: 'int[:]', face_normal: 'float[:, :]', face_center: 'float[:, :]',
                         face_measure: 'float[:]', faces: 'int[:, :]', nodes: 'float[:, :]', phy_faces: 'int[:, :]',
                         node_cellid: 'int[:, :]', phyid_to_faceid: 'int[:]', nb_phy_faces, phy_faces_name, dim):

    ghost_info_size = nb_phy_faces

    # ---- bf_cellid
    bf_cellid = np.zeros(shape=(ghost_info_size, 2), dtype=self.config.int_dtype)
    intersect = np.zeros(shape=2, dtype=self.config.int_dtype)
    self.compute.create_bf_cellid(phy_faces, node_cellid, phyid_to_faceid, cell_faceid, intersect, bf_cellid)

    # Periodic faces (names 11/22/33/44/55/66) are NOT physical boundaries: their
    # partner cell is delivered through the (periodic) halo, so they must not get a
    # ghost. An unvalued periodic ghost would bias the node interpolation of no-BC
    # variables toward 0 (VTK). Mark them invalid so both ghost kernels skip them.
    pn = phy_faces_name
    per = (pn == 11) | (pn == 22) | (pn == 33) | (pn == 44) | (pn == 55) | (pn == 66)
    if np.any(per):
      bf_cellid[per] = -1

    # ---- ghost_info_flt, ghost_info_int
    ghost_info_data_size_flt = 10  # (ghostcenter_x&y&z, gamma, face_center_x&y&z, face_normal_x&y&z)
    ghost_info_data_size_int = 5  # (cell_id, face index inside the cell, face_oldname, cell global id, face_id)
    ghost_info_flt = np.zeros(shape=(ghost_info_size, ghost_info_data_size_flt), dtype=self.config.float_dtype)
    ghost_info_int = np.zeros(shape=(ghost_info_size, ghost_info_data_size_int), dtype=self.config.int_dtype)

    self.compute.create_ghost_info(bf_cellid, cell_center, cell_faceid, cell_loctoglob, faces, nodes, face_oldname,
                              face_normal, face_center, face_measure, ghost_info_int, ghost_info_flt, dim)

    return ghost_info_int, ghost_info_flt

  def create_ghost_tables(self, ghost_info_int: 'int[:, :]', node_cellid: 'int[:, :]', faces: 'int[:, :]', cell_faceid: 'int[:, :]', max_node_phyid, max_cell_phyid):

    max_cell_ghostnid = max_cell_phyid
    nb_cells = len(cell_faceid)
    nb_nodes = len(node_cellid)
    nb_faces = len(faces)

    cell_ghostnid = np.zeros(shape=(nb_cells, max_cell_ghostnid + 1), dtype=self.config.int_dtype)
    node_ghostid = np.zeros(shape=(nb_nodes, max_node_phyid + 1), dtype=self.config.int_dtype)

    ghost_i_visited = np.ones(shape=nb_faces, dtype=self.config.int_dtype) * -1
    self.compute.create_ghost_tables(ghost_info_int, faces, cell_faceid, node_cellid, ghost_i_visited, node_ghostid, cell_ghostnid)

    return (
      node_ghostid,
      cell_ghostnid
    )

  def create_halo_ghost_tables(self, ext_ghost_info_int: 'float[:, :]', node_halophyid: 'int[:]', cell_halophyid: 'int[:]', node_haloid: 'int[:, :]', halo_halosext: 'int[:, :]', max_cell_halophyid, max_node_halophyid, size, nb_nodes, nb_cells):
    # node_halophyid / cell_halophyid are FLAT run-length lists
    # ([id, size, phyid...] repeated), not per-node/per-cell tables -- their
    # length says nothing about the mesh size (it is 0 in serial). The output
    # tables are indexed by local node/cell id, so they must be sized by the
    # mesh counts.
    if size == 1:
      # give size to cell_haloghostnid and node_haloghostid to keep the multiprocessing code as it is
      cell_haloghostid = np.zeros(shape=(nb_cells, 1), dtype=self.config.int_dtype)
      node_haloghostid = np.zeros(shape=(nb_nodes, 1), dtype=self.config.int_dtype)
    else:
      cell_haloghostid = np.zeros(shape=(nb_cells, max_cell_halophyid + 1), dtype=self.config.int_dtype)
      node_haloghostid = np.zeros(shape=(nb_nodes, max_node_halophyid + 1), dtype=self.config.int_dtype)
      # It will also update ext_ghost_info_int[0] from cell_id to haloext of the cell
      self.compute.create_halo_ghost_tables(ext_ghost_info_int, node_halophyid, cell_halophyid, node_haloid, halo_halosext, cell_haloghostid, node_haloghostid)

    return (
      cell_haloghostid,
      node_haloghostid
    )

  def face_gradient_info(self, face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, cell_shift, dim):
    nb_faces = len(faces)

    face_air_diamond = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param1 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param2 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param3 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_param4 = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_f1 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f2 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f3 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)
    face_f4 = np.zeros(shape=(nb_faces, dim), dtype=self.config.float_dtype)

    if dim == 2:
      self.compute.face_gradient_info_2d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_param4, face_f1, face_f2, face_f3, face_f4, cell_shift)
    else:
      self.compute.face_gradient_info_3d(face_cellid, faces, face_to_phyid, ghost_info_flt, face_name, face_normal, cell_center, halo_centvol, face_haloid, nodes, face_air_diamond, face_param1, face_param2, face_param3, face_f1, face_f2, cell_shift)

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

  def fv_face_geometry(self, face_cellid, face_name, face_normal, face_center, face_haloid, cell_center, halo_centvol, cell_shift):
    nb_faces = len(face_normal)

    face_fv_coeff = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corrx = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corry = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_corrz = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    face_fv_weight_left = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    self.compute.fv_face_geometry(face_cellid, face_name, face_normal, face_center, face_haloid, cell_center, halo_centvol, cell_shift, face_fv_coeff, face_fv_corrx, face_fv_corry, face_fv_corrz, face_fv_weight_left)

    return (
      face_fv_coeff,
      face_fv_corrx,
      face_fv_corry,
      face_fv_corrz,
      face_fv_weight_left
    )

  def variables(self, cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, cell_shift, dim):
    nb_nodes = len(nodes)

    node_R_x = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_R_y = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_R_z = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_x = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_y = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_lambda_z = np.zeros(nb_nodes, dtype=self.config.float_dtype)
    node_number = np.zeros(nb_nodes, dtype=self.config.int_dtype)

    if dim == 2:
      self.compute.variables_2d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_lambda_x, node_lambda_y, node_number, cell_shift)
    else:
      self.compute.variables_3d(cell_center, node_cellid, node_haloid, node_ghostid, node_haloghostid, node_periodicid, nodes, node_oldname, ghost_info_flt, ext_ghost_info_flt, halo_centvol, node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number, cell_shift)

    return (
      node_R_x,
      node_R_y,
      node_R_z,
      node_lambda_x,
      node_lambda_y,
      node_lambda_z,
      node_number
    )

  def create_normal_face_of_cell(self, cell_center: 'float[:,:]', face_center: 'float[:,:]', cell_faceid: 'int[:,:]', face_normal: 'float[:,:]', max_cell_faceid):
    nb_cells = len(cell_center)

    cell_nf = np.zeros(shape=(nb_cells, max_cell_faceid, 3), dtype=self.config.float_dtype)
    self.compute.create_normal_face_of_cell(cell_center, face_center, cell_faceid, face_normal, cell_nf)
    return cell_nf

  def dist_ortho_function_2d(self, d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', face_cellid: 'int[:,:]', cell_center: 'float[:,:]', face_center: 'float[:,:]', face_normal: 'float[:,:]', dim):
    nb_faces = len(face_normal)
    face_dist_ortho = np.zeros(shape=nb_faces, dtype=self.config.float_dtype)
    if dim == 2:
      self.compute.dist_ortho_function_2d(d_innerfaces, d_boundaryfaces, face_cellid, cell_center, face_center, face_normal, face_dist_ortho)
    return face_dist_ortho

  def build_periodic_samerank(self, nodes, node_cellid, faces, face_name, face_center, face_cellid, cell_shift, dim):
    # SAME-RANK periodic pairing. For periodic pairs whose BOTH sides live in this
    # subdomain it wires:
    #   * face_cellid[face][1] = partner cell + cell_shift (image across boundary),
    #     so the flux and geometry kernels treat the periodic face like interior;
    #   * node_periodicid[node] = partner node's cells, so the VTK node
    #     interpolation sees both sides (else its least-squares stencil is
    #     one-sided -> singular).
    # Sign of the shift by the boundary a cell touches: x=0/y=0/z=0 (tags
    # 11/44/55) -> +L on that component, x=Lx/y=Ly/z=Lz (tags 22/33/66) -> -L.
    #
    # CROSS-RANK pairs are NOT seen here: the C++ partitioner
    # (handle_periodic_faces) already turned them into halo faces (face_name==10,
    # translated halo_centvol), so they no longer carry tags 11/22/33/44/55/66.
    #
    # The matching itself is done in COMPILED kernels (domain_compute:
    # pair_periodic_faces / match_periodic_nodes / fill_node_periodicid). They are
    # sort-based (no Python dict) and O(#periodic boundary faces/nodes), so this
    # stays cheap even in 3D where those are O(N^2) surface quantities.
    #
    # Lx/Ly/Lz come from the LOCAL node extent, which is EXACT for same-rank
    # pairs: a same-rank pair needs both boundaries of that axis in THIS
    # subdomain, so it spans the full box on that axis (max-min == global length).
    nb_nodes = len(nodes)
    node_periodicid = np.zeros((nb_nodes, 2), dtype=self.config.int_dtype)
    fname = face_name
    if not np.any((fname == 11) | (fname == 22) | (fname == 33) |
                  (fname == 44) | (fname == 55) | (fname == 66)):
      return node_periodicid

    nmin = np.array([nodes[:, 0].min(), nodes[:, 1].min(),
                     nodes[:, 2].min()], dtype=self.config.float_dtype)
    nmax = np.array([nodes[:, 0].max(), nodes[:, 1].max(),
                     nodes[:, 2].max()], dtype=self.config.float_dtype)
    dtol = 1e-6 * float(np.max(nmax - nmin))   # transverse-match tolerance
    Lx = float(nmax[0] - nmin[0])
    Ly = float(nmax[1] - nmin[1])
    Lz = float(nmax[2] - nmin[2])
    t2 = 2 if dim == 3 else -1

    # faces: (name_lo, name_hi, taxis0, taxis1, shift_axis, L)
    fdirs = [(11, 22, 1, t2, 0, Lx), (44, 33, 0, t2, 1, Ly)]
    if dim == 3:
      fdirs.append((55, 66, 0, 1, 2, Lz))
    for (name_lo, name_hi, t0, t1, sax, L) in fdirs:
      r = self.compute.pair_periodic_faces(face_name, face_center,
                                      face_cellid, cell_shift, nmin,
                                      name_lo, name_hi, t0, t1, sax, L, dtol)
      if r < 0:
        raise ValueError(
          "same-rank periodic face pairing failed (code %d) for tags %d/%d: a "
          "periodic face has no local partner (cross-rank should already be a "
          "halo) or the two boundaries are non-conforming." % (r, name_lo, name_hi))

    # nodes: per-node periodic-boundary bitmask (from the periodic faces), then
    # accumulate partner cells per periodic axis into node_periodicid. An edge or
    # corner node carries several bits, so it collects partners from EVERY
    # direction it touches -- fixing the 3D one-sided-stencil div-by-zero that a
    # single node_oldname tag caused. Width 3*node_cellid gives room for up to 3
    # directions of partners.
    node_bits = np.zeros(nb_nodes, dtype=self.config.int_dtype)
    self.compute.node_periodic_bits(faces, face_name, node_bits)
    npid = np.zeros((nb_nodes, 3 * node_cellid.shape[1]),
                    dtype=self.config.int_dtype)
    node_fill = np.zeros(nb_nodes, dtype=self.config.int_dtype)
    # bits: 1=x-lo(11) 2=x-hi(22) 4=y-hi(33) 8=y-lo(44) 16=z-lo(55) 32=z-hi(66)
    self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                               node_fill, nmin, 1, 2, 1, t2, dtol)   # x -> (y[,z])
    self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                               node_fill, nmin, 8, 4, 0, t2, dtol)   # y -> (x[,z])
    if dim == 3:
      self.compute.accum_periodic_dir(node_bits, nodes, node_cellid, npid,
                                 node_fill, nmin, 16, 32, 0, 1, dtol)  # z -> (x,y)
    if np.any(node_fill > 0):
      return npid
    return node_periodicid




class VariableCompute:
  """Device-agnostic entry points for the per-variable kernels.

  The compiled kernels take numpy arrays (CPU build) or CuPy arrays (CUDA
  build); they never see a ManapyArray. Each kernel is exposed here as a
  *static* wrapper that unwraps every argument through the ManapyArray
  interface with the intent the kernel actually has for it:

      r  -> cpu_r  / gpu_r    read-only:  sync in if stale, other side stays valid
      rw -> cpu_rw / gpu_rw   read-write: sync in if stale, other side invalidated
      w  -> cpu_w  / gpu_w    write-only: NO transfer, other side invalidated

  `w` is only legal when the kernel writes EVERY element -- it hands out an
  uninitialised buffer otherwise. Kernels looping over the full extent of their
  output (facetocell, center_to_vertex, cell_gradient, the limiters) qualify;
  kernels writing through a gather list of face ids (celltoface,
  face_gradient) touch only part of the array and must use `rw`.

  `__init__` resolves the CPU/CUDA and 2D/3D kernel plus the matching accessor
  triple once and binds them, so callers keep calling e.g.
  ``compute.cell_gradient(...)`` with the same positional argument list as
  before. All array arguments must be ManapyArray.

  The binding is done with `functools.partial`, which freezes the leading
  arguments of a function and returns a callable taking the rest. Each wrapper
  is declared as ``(kernel, acc, <kernel args...>)`` and

      self.facetocell = partial(VariableCompute.facetocell, k_facetocell, acc)

  pins the first two, so `self.facetocell` behaves as if its signature were
  ``(u_face, u_c, cell_faceid, dim)`` -- the device/dim choice is resolved once
  here instead of on every call. A closure
  (``lambda *a: VariableCompute.facetocell(k, acc, *a)``) would do the same,
  but `partial` prepends the frozen arguments in C (no extra Python frame) and
  is picklable, which a lambda is not -- and manapy runs under MPI.

  The wrappers are positional-only in practice: the 2D and 3D bindings do not
  always agree on parameter names (``face_halofid`` vs ``face_haloid``,
  ``d_periodicfaces`` vs ``d_periodicboundaryfaces``), so a single wrapper
  serves both and keyword calls must not be used.
  """

  def __init__(self, config: ManapyConfig, dim: int):
    self.config = config
    self.dim = dim
    self.compute = _Compute(config)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_facetocell = self.compute.facetocell_cuda
      k_celltoface = self.compute.celltoface_cuda
      if dim == 2:
        k_interp = self.compute.center_to_vertex_2d_cuda
        k_face_gradient = self.compute.face_gradient_2d_cuda
        k_cell_gradient = self.compute.cell_gradient_2d_cuda
        k_barthlimiter = self.compute.barthlimiter_2d_cuda
        k_vanalbadalimiter = self.compute.vanalbadalimiter_2d_cuda
      else:
        k_interp = self.compute.center_to_vertex_3d_cuda
        k_face_gradient = self.compute.face_gradient_3d_cuda
        k_cell_gradient = self.compute.cell_gradient_3d_cuda
        k_barthlimiter = self.compute.barthlimiter_3d_cuda
        k_vanalbadalimiter = self.compute.vanalbadalimiter_3d_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_facetocell = self.compute.facetocell
      k_celltoface = self.compute.celltoface
      if dim == 2:
        k_interp = self.compute.center_to_vertex_2d
        k_face_gradient = self.compute.face_gradient_2d
        k_cell_gradient = self.compute.cell_gradient_2d
        k_barthlimiter = self.compute.barthlimiter_2d
        k_vanalbadalimiter = self.compute.vanalbadalimiter_2d
      else:
        k_interp = self.compute.center_to_vertex_3d
        k_face_gradient = self.compute.face_gradient_3d
        k_cell_gradient = self.compute.cell_gradient_3d
        k_barthlimiter = self.compute.barthlimiter_3d
        k_vanalbadalimiter = self.compute.vanalbadalimiter_3d

    self.facetocell = partial(VariableCompute.facetocell, k_facetocell, acc)
    self.celltoface = partial(VariableCompute.celltoface, k_celltoface, acc)
    self.interp = partial(VariableCompute.interp, k_interp, acc)
    self.face_gradient = partial(VariableCompute.face_gradient, k_face_gradient, acc)
    self.cell_gradient = partial(VariableCompute.cell_gradient, k_cell_gradient, acc)
    # Both limiters have the same signature and the same read/write intents.
    self.barthlimiter = partial(VariableCompute.limiter, k_barthlimiter, acc)
    self.vanalbadalimiter = partial(VariableCompute.limiter, k_vanalbadalimiter, acc)

  # ------------------------------------------------------------------ kernels

  @staticmethod
  def facetocell(kernel, acc, u_face, u_c, cell_faceid, dim):
    """Face field -> cell field. `u_c` is written for every cell."""
    r, rw, w = acc
    kernel(r(u_face), w(u_c), r(cell_faceid), dim)

  @staticmethod
  def celltoface(kernel, acc, u_cell, u_face, u_ghost, u_halo, face_cellid,
                 face_halofid, d_innerfaces, d_boundaryfaces, d_halofaces):
    """Cell field -> face field. `u_face` is written only at the gathered face
    ids (inner / halo / boundary), so it is read-write, not write-only."""
    r, rw, w = acc
    kernel(
      r(u_cell), rw(u_face), r(u_ghost), r(u_halo), r(face_cellid),
      r(face_halofid), r(d_innerfaces), r(d_boundaryfaces), r(d_halofaces)
    )

  @staticmethod
  def interp(kernel, acc, w_c, w_ghost, w_halo, w_haloghost, cell_center,
             halo_centvol, node_cellid, ghost_info_flt, ghost_ext_info_flt,
             node_ghostid, node_haloghostid, node_periodicid, node_halonid,
             nodes, node_oldname, node_R_x, node_R_y, node_R_z, node_lambda_x,
             node_lambda_y, node_lambda_z, node_number, cell_shift, w_n,
             ghost_faceid):
    """center_to_vertex_2d/3d: cell field -> node field. `w_n` is written for
    every node."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_haloghost), r(cell_center),
      r(halo_centvol), r(node_cellid), r(ghost_info_flt), r(ghost_ext_info_flt),
      r(node_ghostid), r(node_haloghostid), r(node_periodicid), r(node_halonid),
      r(nodes), r(node_oldname), r(node_R_x), r(node_R_y), r(node_R_z),
      r(node_lambda_x), r(node_lambda_y), r(node_lambda_z), r(node_number),
      r(cell_shift), w(w_n), r(ghost_faceid)
    )

  @staticmethod
  def cell_gradient(kernel, acc, w_c, w_ghost, w_halo, w_haloghost, cell_center,
                    cell_cellnid, ghost_info_flt, ghost_ext_info_flt,
                    cell_ghostnid, cell_haloghostnid, cell_halonid, cells,
                    cell_periodicfid, node_periodicid, node_oldname,
                    halo_centvol, cell_shift, w_x, w_y, w_z, ghost_faceid):
    """cell_gradient_2d/3d: least-squares gradient at the cell centres.
    `w_x`/`w_y`/`w_z` are written for every cell (the 2D kernel sets w_z to 0
    explicitly), so all three are write-only."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_haloghost), r(cell_center),
      r(cell_cellnid), r(ghost_info_flt), r(ghost_ext_info_flt),
      r(cell_ghostnid), r(cell_haloghostnid), r(cell_halonid), r(cells),
      r(cell_periodicfid), r(node_periodicid), r(node_oldname),
      r(halo_centvol), r(cell_shift), w(w_x), w(w_y), w(w_z), r(ghost_faceid)
    )

  @staticmethod
  def face_gradient(kernel, acc, w_c, w_ghost, w_halo, w_node, face_cellid,
                    faces, face_halofid, face_air_diamond, face_normal, face_f1,
                    face_f2, face_f3, face_f4, wx_face, wy_face, wz_face,
                    d_innerfaces, d_halofaces, dirichletfaces, neumann,
                    d_periodicfaces):
    """face_gradient_2d/3d: gradient at the face midpoints. The outputs are
    written only at the gathered face ids -- and the 2D kernel does not touch
    `wz_face` at all -- so all three are read-write."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_node), r(face_cellid), r(faces),
      r(face_halofid), r(face_air_diamond), r(face_normal), r(face_f1),
      r(face_f2), r(face_f3), r(face_f4), rw(wx_face), rw(wy_face),
      rw(wz_face), r(d_innerfaces), r(d_halofaces), r(dirichletfaces),
      r(neumann), r(d_periodicfaces)
    )

  @staticmethod
  def limiter(kernel, acc, w_c, w_ghost, w_halo, w_x, w_y, w_z, psi,
              face_cellid, cell_faceid, face_name, face_haloid, cell_center,
              face_center):
    """barthlimiter_2d/3d and vanalbadalimiter_2d/3d: same signature and same
    intents. `psi` is written for every cell (the 2D kernels ignore `w_z`)."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_x), r(w_y), r(w_z), w(psi),
      r(face_cellid), r(cell_faceid), r(face_name), r(face_haloid),
      r(cell_center), r(face_center)
    )


class BoundaryCompute:
  def __init__(self, config: ManapyConfig, dim: int, BCtype: str):
    self.config = config
    self.dim = dim
    self.BCtype = BCtype
    self.compute = _Compute(config)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      if BCtype == "dirichlet":
        k_ghost = self.compute.ghost_value_dirichlet_cuda
        k_haloghost = self.compute.haloghost_value_dirichlet_cuda
      elif BCtype == "neumann" or BCtype == "periodic":
        k_ghost = self.compute.ghost_value_neumann_cuda
        k_haloghost = self.compute.haloghost_value_neumann_cuda
      elif BCtype == "neumannNH":
        k_ghost = self.compute.ghost_value_neumannNH_cuda
        k_haloghost = self.compute.haloghost_value_neumannNH_cuda
      elif BCtype == "nonslip":
        k_ghost = self.compute.ghost_value_nonslip_cuda
        k_haloghost = self.compute.haloghost_value_nonslip_cuda
      elif BCtype == "slip":
        if dim == 2:
          k_ghost = self.compute.ghost_value_slip_2d_cuda
          k_haloghost = self.compute.haloghost_value_slip_2d_cuda
        else:
          k_ghost = self.compute.ghost_value_slip_3d_cuda
          k_haloghost = self.compute.haloghost_value_slip_3d_cuda
      else:
        raise ValueError(f"unknown BCtype: {BCtype}")
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      if BCtype == "dirichlet":
        k_ghost = self.compute.ghost_value_dirichlet
        k_haloghost = self.compute.haloghost_value_dirichlet
      elif BCtype == "neumann" or BCtype == "periodic":
        k_ghost = self.compute.ghost_value_neumann
        k_haloghost = self.compute.haloghost_value_neumann
      elif BCtype == "neumannNH":
        k_ghost = self.compute.ghost_value_neumannNH
        k_haloghost = self.compute.haloghost_value_neumannNH
      elif BCtype == "nonslip":
        k_ghost = self.compute.ghost_value_nonslip
        k_haloghost = self.compute.haloghost_value_nonslip
      elif BCtype == "slip":
        if dim == 2:
          k_ghost = self.compute.ghost_value_slip_2d
          k_haloghost = self.compute.haloghost_value_slip_2d
        else:
          k_ghost = self.compute.ghost_value_slip_3d
          k_haloghost = self.compute.haloghost_value_slip_3d
      else:
        raise ValueError(f"unknown BCtype: {BCtype}")

    # The four scalar conditions share one signature, so one wrapper serves
    # them all. Slip does not: it takes 2 or 3 velocity components, so it
    # needs its own wrapper per dimension, not just its own kernel.
    if BCtype != "slip":
      w_ghost = BoundaryCompute.ghost_value
      w_haloghost = BoundaryCompute.haloghost_value
    elif dim == 2:
      w_ghost = BoundaryCompute.slip_ghost_2d
      w_haloghost = BoundaryCompute.slip_haloghost_2d
    else:
      w_ghost = BoundaryCompute.slip_ghost_3d
      w_haloghost = BoundaryCompute.slip_haloghost_3d

    self.ghost = partial(w_ghost, k_ghost, acc)
    self.haloghost = partial(w_haloghost, k_haloghost, acc)

  # ------------------------------------------------------------------ kernels

  @staticmethod
  def ghost_value(kernel, acc, value, w_ghost, face_cellid, bc_faces, cst,
                  face_dist_ortho):
    """ghost_value_dirichlet/_neumann/_neumannNH/_nonslip: set the ghost value
    behind every face of `bc_faces`.

    `value` is the prescribed per-face value for dirichlet and the cell field
    `w_c` for the other kinds -- the same slot in every signature. `cst` and
    `face_dist_ortho` are read by neumannNH only, `face_cellid` by everything
    except dirichlet; the unused ones stay in the signature for parity."""
    r, rw, w = acc
    kernel(
      r(value), rw(w_ghost), r(face_cellid), r(bc_faces), r(cst),
      r(face_dist_ortho)
    )

  @staticmethod
  def haloghost_value(kernel, acc, value, w_haloghost, node_haloghostid,
                      ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                      d_halonodes, cst):
    """haloghost_value_dirichlet/_neumann/_neumannNH/_nonslip: set the halo
    ghosts tagged `BCindex` that hang off a node of `d_halonodes`.

    `value` is indexed per halo ghost for dirichlet and is the halo cell field
    `w_halo` for the other kinds -- two different index spaces (sizehaloghost
    vs nbhalos) in the same slot. `BCindex` is a scalar, not an array."""
    r, rw, w = acc
    kernel(
      r(value), rw(w_haloghost), r(node_haloghostid), r(ghost_ext_info_int),
      r(ghost_ext_info_flt), BCindex, r(d_halonodes), r(cst)
    )

  @staticmethod
  def slip_ghost_2d(kernel, acc, u_c, v_c, u_ghost, v_ghost, face_cellid,
                    bc_faces, normal):
    """ghost_value_slip_2d: free-slip reflection of the velocity behind every
    face of `bc_faces`. Coupled: both components are needed together."""
    r, rw, w = acc
    kernel(
      r(u_c), r(v_c), rw(u_ghost), rw(v_ghost), r(face_cellid), r(bc_faces),
      r(normal)
    )

  @staticmethod
  def slip_ghost_3d(kernel, acc, u_c, v_c, w_c, u_ghost, v_ghost, w_ghost,
                    face_cellid, bc_faces, normal):
    """ghost_value_slip_3d: 3D counterpart of slip_ghost_2d."""
    r, rw, w = acc
    kernel(
      r(u_c), r(v_c), r(w_c), rw(u_ghost), rw(v_ghost), rw(w_ghost),
      r(face_cellid), r(bc_faces), r(normal)
    )

  @staticmethod
  def slip_haloghost_2d(kernel, acc, u_halo, v_halo, u_haloghost, v_haloghost,
                        node_haloghostid, ghost_ext_info_int,
                        ghost_ext_info_flt, BCindex, d_halonodes):
    """haloghost_value_slip_2d: free-slip reflection on the halo ghosts tagged
    `BCindex`. No `cst` argument, unlike the scalar haloghost kernels."""
    r, rw, w = acc
    kernel(
      r(u_halo), r(v_halo), rw(u_haloghost), rw(v_haloghost),
      r(node_haloghostid), r(ghost_ext_info_int), r(ghost_ext_info_flt),
      BCindex, r(d_halonodes)
    )

  @staticmethod
  def slip_haloghost_3d(kernel, acc, u_halo, v_halo, w_halo, u_haloghost,
                        v_haloghost, w_haloghost, node_haloghostid,
                        ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                        d_halonodes):
    """haloghost_value_slip_3d: 3D counterpart of slip_haloghost_2d."""
    r, rw, w = acc
    kernel(
      r(u_halo), r(v_halo), r(w_halo), rw(u_haloghost), rw(v_haloghost),
      rw(w_haloghost), r(node_haloghostid), r(ghost_ext_info_int),
      r(ghost_ext_info_flt), BCindex, r(d_halonodes)
    )


class AdvectionSolverCompute:
  """Device-agnostic entry points for the advection-solver kernels.

  Same contract as `VariableCompute` and `BoundaryCompute`: the compiled
  kernels take numpy arrays (CPU build) or CuPy arrays (CUDA build) and never
  see a ManapyArray:

      r  -> cpu_r  / gpu_r    read-only:  sync in if stale, other side stays valid
      rw -> cpu_rw / gpu_rw   read-write: sync in if stale, other side invalidated
      w  -> cpu_w  / gpu_w    write-only: NO transfer, other side invalidated

  `__init__` resolves the CPU/CUDA and 2D/3D kernel plus the matching accessor
  triple once and binds them with `functools.partial` (see `VariableCompute`
  for why partial rather than a lambda), so callers keep the same positional
  argument list the raw kernels take.

  The wrappers are positional-only in practice: one wrapper serves both the 2D
  and the 3D convective kernel, so keyword calls must not be used. Scalars
  (`cfl`, `dtime`, `order`, `scheme`, `dim`) are passed straight through --
  only the array arguments must be ManapyArray.

  `update_new_value` and the two `initialisation_gaussian_*` are not advec
  kernels: they live in solvers.utils and are shared by every solver, and are
  exposed here because the advection solver's setup and time loop need them.
  The Gaussian ones are CPU-only and so always run through the CPU accessors
  (see `__init__`).
  """

  def __init__(self, config: ManapyConfig, dim: int):
    self.config = config
    self.dim = dim
    self.compute = _Compute(config)

    if dim not in (2, 3):
      raise ValueError(f"dim must be 2 or 3, got {dim}")

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_time_step = self.compute.advec_time_step_cuda
      k_update_new_value = self.compute.update_new_value_cuda
      if dim == 2:
        k_convective = self.compute.advec_explicitscheme_convective_2d_cuda
      else:
        k_convective = self.compute.advec_explicitscheme_convective_3d_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_time_step = self.compute.advec_time_step
      k_update_new_value = self.compute.update_new_value
      if dim == 2:
        k_convective = self.compute.advec_explicitscheme_convective_2d
      else:
        k_convective = self.compute.advec_explicitscheme_convective_3d

    # The Gaussian initial conditions are CPU-only -- there is no _cuda kernel
    # to pick, so they always get the CPU accessor triple, whatever the device.
    # That is not a fallback but the correct handling: `cpu_w` hands out the
    # host buffer without a transfer and marks the device copy stale, so the
    # kernel fills the field on the host and the first `gpu_r` after it syncs
    # the result up. Only `cell_center` costs a device->host copy under CUDA,
    # once, at setup.
    cpu_acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)

    self.explicitscheme_convective = partial(
      AdvectionSolverCompute.explicitscheme_convective, k_convective, acc)
    self.time_step = partial(AdvectionSolverCompute.time_step, k_time_step, acc)
    self.update_new_value = partial(
      AdvectionSolverCompute.update_new_value, k_update_new_value, acc)
    self.initialisation_gaussian_2d = partial(
      AdvectionSolverCompute.initialisation_gaussian_2d,
      self.compute.initialisation_gaussian_2d, cpu_acc)
    self.initialisation_gaussian_3d = partial(
      AdvectionSolverCompute.initialisation_gaussian_3d,
      self.compute.initialisation_gaussian_3d, cpu_acc)

  # ------------------------------------------------------------------ kernels

  @staticmethod
  def explicitscheme_convective(kernel, acc, rez_w, w_c, w_ghost, w_halo,
                                u_face, v_face, w_face, w_x, w_y, w_z, wx_halo,
                                wy_halo, wz_halo, psi, psi_halo, cell_center,
                                face_center, halo_centvol, face_cellid,
                                face_normal, face_haloid, face_name,
                                d_innerfaces, d_halofaces, d_boundaryfaces,
                                d_periodicboundaryfaces, cell_shift, order,
                                scheme):
    """explicitscheme_convective_2d/3d: explicit finite-volume convective
    residual. Both kernels zero `rez_w` over every cell before scattering the
    face fluxes into it, so it is write-only. `w_z` / `wz_halo` are read by the
    3D kernel only; they stay in the signature for parity."""
    r, rw, w = acc
    kernel(
      w(rez_w), r(w_c), r(w_ghost), r(w_halo), r(u_face), r(v_face), r(w_face),
      r(w_x), r(w_y), r(w_z), r(wx_halo), r(wy_halo), r(wz_halo), r(psi),
      r(psi_halo), r(cell_center), r(face_center), r(halo_centvol),
      r(face_cellid), r(face_normal), r(face_haloid), r(face_name),
      r(d_innerfaces), r(d_halofaces), r(d_boundaryfaces),
      r(d_periodicboundaryfaces), r(cell_shift), order, scheme
    )

  @staticmethod
  def time_step(kernel, acc, u, v, w_, cfl, face_normal, face_measure,
                cell_volume, cell_faceid, dim):
    """Explicit CFL time step: min over the cells of cfl * volume / sum(|u.n|).
    Reads only, and returns the time step as a Python float -- the caller still
    has to reduce it across ranks. `face_measure` and `dim` are unused by the
    computation and kept for signature parity."""
    r, rw, w = acc
    return kernel(
      r(u), r(v), r(w_), cfl, r(face_normal), r(face_measure), r(cell_volume),
      r(cell_faceid), dim
    )

  @staticmethod
  def update_new_value(kernel, acc, ne_c, rez_ne, dissip_ne, src_ne, dtime,
                       cell_volume):
    """solvers.utils.update_new_value: forward-Euler update of a cell field,
    ne_c += dtime * ((rez + dissip) / volume + src). `ne_c` is accumulated
    into, not overwritten, so it is read-write."""
    r, rw, w = acc
    kernel(
      rw(ne_c), r(rez_ne), r(dissip_ne), r(src_ne), dtime, r(cell_volume)
    )

  @staticmethod
  def initialisation_gaussian_2d(kernel, acc, ne, u, v, P, cell_center, Pinit):
    """solvers.utils.initialisation_gaussian_2d: Gaussian bump initial
    condition -- ne = Gaussian centred at (0.2, 0.2), u = v = 0 and
    P = Pinit * (0.5 - x).

    The kernel loops over every cell of `cell_center` and assigns (never
    accumulates) each output, so `ne`, `u`, `v` and `P` are all write-only.
    They must be at least as long as `cell_center` -- the loop bound comes from
    `cell_center`, not from the outputs."""
    r, rw, w = acc
    kernel(w(ne), w(u), w(v), w(P), r(cell_center), Pinit)

  @staticmethod
  def initialisation_gaussian_3d(kernel, acc, ne, u, v, w_, P, cell_center,
                                 Pinit):
    """solvers.utils.initialisation_gaussian_3d: 3D counterpart of
    initialisation_gaussian_2d -- Gaussian centred at (0.2, 0.25, 0.45),
    u = v = w = 0 and P = Pinit * (0.5 - x). Same write-only outputs, plus the
    z velocity (named `w_` here so it does not shadow the `w` accessor)."""
    r, rw, w = acc
    kernel(w(ne), w(u), w(v), w(w_), w(P), r(cell_center), Pinit)
