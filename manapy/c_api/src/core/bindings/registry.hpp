#pragma once

#include "manapy_compute_types.hpp"

// Each kernel's binding TU exposes one register_* function that adds its Python
// entry points to the module. bindings/module.cpp defines NB_MODULE(_core) and
// calls them all, so kernels stay in separate translation units instead of one
// growing module file.
void register_cell_gradient_2d(nb::module_ &m);
void register_center_to_vertex_2d(nb::module_ &m);
void register_face_gradient_2d(nb::module_ &m);
void register_barthlimiter_2d(nb::module_ &m);
void register_vanalbadalimiter_2d(nb::module_ &m);
void register_face_gradient_3d(nb::module_ &m);
void register_cell_gradient_3d(nb::module_ &m);
void register_center_to_vertex_3d(nb::module_ &m);
void register_barthlimiter_3d(nb::module_ &m);
void register_vanalbadalimiter_3d(nb::module_ &m);
void register_facetocell(nb::module_ &m);
void register_celltoface(nb::module_ &m);
