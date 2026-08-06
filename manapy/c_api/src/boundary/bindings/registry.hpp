#pragma once

#include "manapy_compute_types.hpp"

// Each kernel's binding TU exposes one register_* function that adds its Python
// entry points to the module. bindings/module.cpp defines NB_MODULE(_core) and
// calls them all, so kernels stay in separate translation units instead of one
// growing module file.
//
// The two scalar families share a binding TU each (their four variants differ
// only by a compile-time tag), so one register_* covers all four: eight m.def
// entry points, CPU and _cuda.
void register_ghost_value(nb::module_ &m);
void register_haloghost_value(nb::module_ &m);
void register_ghost_value_slip_2d(nb::module_ &m);
void register_ghost_value_slip_3d(nb::module_ &m);
void register_haloghost_value_slip_2d(nb::module_ &m);
void register_haloghost_value_slip_3d(nb::module_ &m);
