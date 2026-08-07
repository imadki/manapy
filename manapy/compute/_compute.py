from manapy.backends.config import ManapyConfig
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

_CACHED = {
  "float64": {
    "int32": None,
    "int64": None,
  },
  "float32": {
    "int32": None,
    "int64": None,
  }
}


class _Compute:
  """
  All manapy_compute_xx_xx classes have the same functions and structure.
  All CPU and GPU functions are exposed from the C API here.
  The configuration must be based only on floatxx and intxx.
  The implementations are located in the c_api/src folder.
  This class is private and used only by the compute folder.
  Only this class should be cached.
  Use the same comments structure.
  """
  def __init__(self, config: ManapyConfig):
    self.config = config
    self.manapy_compute : "manapy_compute_64_32" = _manapy_compute[self.config.float_precision][self.config.int_precision]

    # ---------------------------------------------------------------------
    # Partitioning compute -> DomainCompute class
    # ---------------------------------------------------------------------
    self.make_n_part_graph_k_way = self.manapy_compute.partitioning.make_n_part_graph_k_way
    self.make_n_part_mesh_dual = self.manapy_compute.partitioning.make_n_part_mesh_dual
    self.make_n_part_mesh_nodal = self.manapy_compute.partitioning.make_n_part_mesh_nodal
    self.create_local_domains = self.manapy_compute.partitioning.create_local_domains

    # ---------------------------------------------------------------------
    # Domain compute -> DomainCompute class
    # ---------------------------------------------------------------------
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


    # ---------------------------------------------------------------------
    # Variable compute -> Variable compute class
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Boundary compute -> Boundary compute class
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Solver utilities (common to every solver)
    # ---------------------------------------------------------------------
    # Cpu (see src/solvers/headers/utils/utils_compute.hpp)
    self.initialisation_gaussian_2d = self.manapy_compute.solvers.utils.initialisation_gaussian_2d # Cpu only
    self.initialisation_gaussian_3d = self.manapy_compute.solvers.utils.initialisation_gaussian_3d # Cpu only
    self.update_new_value = self.manapy_compute.solvers.utils.update_new_value
    # Gpu
    self.update_new_value_cuda = self.manapy_compute.solvers.utils.update_new_value_cuda

    # ---------------------------------------------------------------------
    # Advection compute -> AdvectionCompute class
    # ---------------------------------------------------------------------
    # Advection Cpu
    self.advec_explicitscheme_convective_2d = self.manapy_compute.solvers.advec.explicitscheme_convective_2d
    self.advec_explicitscheme_convective_3d = self.manapy_compute.solvers.advec.explicitscheme_convective_3d
    self.advec_time_step = self.manapy_compute.solvers.advec.time_step
    # Advection Gpu
    self.advec_explicitscheme_convective_2d_cuda = self.manapy_compute.solvers.advec.explicitscheme_convective_2d_cuda
    self.advec_explicitscheme_convective_3d_cuda = self.manapy_compute.solvers.advec.explicitscheme_convective_3d_cuda
    self.advec_time_step_cuda = self.manapy_compute.solvers.advec.time_step_cuda

    # ---------------------------------------------------------------------
    # Advection-diffusion compute -> AdvectionDiffusionSolverCompute class
    # ---------------------------------------------------------------------
    # Advection-diffusion Cpu
    self.advecdiff_explicitscheme_convective_2d = self.manapy_compute.solvers.advecdiff.explicitscheme_convective_2d
    self.advecdiff_explicitscheme_convective_3d = self.manapy_compute.solvers.advecdiff.explicitscheme_convective_3d
    self.advecdiff_explicitscheme_dissipative = self.manapy_compute.solvers.advecdiff.explicitscheme_dissipative
    self.advecdiff_time_step = self.manapy_compute.solvers.advecdiff.time_step
    # Advection-diffusion Gpu
    self.advecdiff_explicitscheme_convective_2d_cuda = self.manapy_compute.solvers.advecdiff.explicitscheme_convective_2d_cuda
    self.advecdiff_explicitscheme_convective_3d_cuda = self.manapy_compute.solvers.advecdiff.explicitscheme_convective_3d_cuda
    self.advecdiff_explicitscheme_dissipative_cuda = self.manapy_compute.solvers.advecdiff.explicitscheme_dissipative_cuda
    self.advecdiff_time_step_cuda = self.manapy_compute.solvers.advecdiff.time_step_cuda

    # ---------------------------------------------------------------------
    # Diffusion compute -> DiffusionSolverCompute class
    # ---------------------------------------------------------------------
    # Diffusion Cpu
    self.diffusion_explicitscheme_dissipative = self.manapy_compute.solvers.diffusion.explicitscheme_dissipative
    self.diffusion_time_step = self.manapy_compute.solvers.diffusion.time_step
    # Diffusion Gpu
    self.diffusion_explicitscheme_dissipative_cuda = self.manapy_compute.solvers.diffusion.explicitscheme_dissipative_cuda
    self.diffusion_time_step_cuda = self.manapy_compute.solvers.diffusion.time_step_cuda


  @staticmethod
  def getComputeInstance(config: ManapyConfig) -> _Compute:
    cached = _CACHED[config.float_precision][config.int_precision]
    if cached is not None:
      return cached
    instance = _Compute(config)
    _CACHED[config.float_precision][config.int_precision] = instance
    return instance