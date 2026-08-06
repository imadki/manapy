// _core module entry point for the manapy_domain_<float bits>_<int bits>
// packages (mesh connectivity/geometry kernels ported from
// src/domain/to_convert.py). Separate from src/core's _core module — see
// "Open decisions" in src/domain/Steps.md. Compiled four times, once per
// precision pair, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS
// selecting real_t/index_t.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy domain (mesh connectivity/geometry) kernels compiled for "
            "float" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
            MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_count_max_node_cellid(m);
  register_create_node_cellid(m);
  register_get_cell_nb_phyid(m);
  register_count_max_cell_cellnid(m);
  register_create_cell_cellnid(m);

  register_create_info(m);

  register_compute_face_info_2d(m);
  register_compute_face_info_3d(m);

  register_create_bf_cellid(m);
  register_create_ghost_info(m);
  register_create_ghost_tables(m);
  register_count_max_bcell_halophyid(m);
  register_create_bcell_halophyid(m);
  register_get_max_b_ncellid(m);
  register_create_b_ncellid(m);

  register_create_halo_ghost_tables(m);

  register_create_cellfid(m);

  register_define_node_oldname(m);
  register_define_face_name(m);

  register_create_halo_cells(m);

  register_face_gradient_info_2d(m);
  register_face_gradient_info_3d(m);

  register_fv_face_geometry(m);

  register_variables_2d(m);
  register_variables_3d(m);

  register_create_normal_face_of_cell(m);
  register_dist_ortho_function_2d(m);

  register_pair_periodic_faces(m);
  register_node_periodic_bits(m);
  register_accum_periodic_dir(m);

  register_compute_cell_center_volume(m);
}
