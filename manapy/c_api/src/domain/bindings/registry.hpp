#pragma once

#include "manapy_compute_types.hpp"

// Each domain kernel's binding TU exposes one register_* function that adds
// its Python entry points to the module; bindings/module.cpp defines
// NB_MODULE(_core) and calls them all.

// Batch 1: node/cell connectivity
void register_count_max_node_cellid(nb::module_ &m);
void register_create_node_cellid(nb::module_ &m);
void register_get_cell_nb_phyid(nb::module_ &m);
void register_count_max_cell_cellnid(nb::module_ &m);
void register_create_cell_cellnid(nb::module_ &m);

// Batch 2: face/cell topology
void register_create_info(nb::module_ &m);

// Batch 3: 3D face geometry
void register_compute_face_info_2d(nb::module_ &m);
void register_compute_face_info_3d(nb::module_ &m);

// Batch 4: ghost / boundary cell tables
void register_create_bf_cellid(nb::module_ &m);
void register_create_ghost_info(nb::module_ &m);
void register_create_ghost_tables(nb::module_ &m);
void register_count_max_bcell_halophyid(nb::module_ &m);
void register_create_bcell_halophyid(nb::module_ &m);
void register_get_max_b_ncellid(nb::module_ &m);
void register_create_b_ncellid(nb::module_ &m);

// Batch 5: halo ghost tables
void register_create_halo_ghost_tables(nb::module_ &m);

// Batch 6: parallel cell-cell connectivity
void register_create_cellfid(nb::module_ &m);

// Batch 7: face naming
void register_define_node_oldname(nb::module_ &m);
void register_define_face_name(nb::module_ &m);

// Batch 8: halo cells
void register_create_halo_cells(nb::module_ &m);

// Batch 9: face-gradient diamond geometry
void register_face_gradient_info_2d(nb::module_ &m);
void register_face_gradient_info_3d(nb::module_ &m);

// Batch 10: FV face geometry
void register_fv_face_geometry(nb::module_ &m);

// Batch 11: node-based least-squares variables
void register_variables_2d(nb::module_ &m);
void register_variables_3d(nb::module_ &m);

// Batch 12: misc geometry
void register_create_normal_face_of_cell(nb::module_ &m);
void register_dist_ortho_function_2d(nb::module_ &m);

// Post-port addition: periodic boundary connectivity
void register_pair_periodic_faces(nb::module_ &m);
void register_node_periodic_bits(nb::module_ &m);
void register_accum_periodic_dir(nb::module_ &m);

// Ported from the manapy c_api: cell centroids and areas/volumes. Registers
// both compute_cell_center_area_2d and compute_cell_center_volume_3d.
void register_compute_cell_center_volume(nb::module_ &m);
