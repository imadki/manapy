#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Declarations for the solver utility kernels: helpers common to all solvers
// (advec, advecDiff, ...) rather than to any one of them, shipped as the
// manapy_compute_<float bits>_<int bits>.solvers.utils submodule.
//
// Convention: utils is a CPU + GPU module. Unless a declaration is explicitly
// commented "CPU-only", it has a matching launch_<kernel> in utils_compute.cuh.
// The Gaussian initial-condition kernels below are CPU-only (marked as such);
// update_new_value has a GPU counterpart.

// CPU-only. Gaussian bump initial condition on a 2D mesh (translation of
// _initialisation_gaussian_2d): ne gets a Gaussian centred at (0.2, 0.2), the
// velocity (u, v) is zeroed and P is set to Pinit * (0.5 - x). All output
// arrays are written in place, indexed by cell.
void initialisation_gaussian_2d(ArrayView<real_t, 1> ne, ArrayView<real_t, 1> u,
                                ArrayView<real_t, 1> v, ArrayView<real_t, 1> P,
                                ArrayView<const real_t, 2> cell_center,
                                real_t Pinit);

// CPU-only. Gaussian bump initial condition on a 3D mesh (translation of
// _initialisation_gaussian_3d): ne gets a Gaussian centred at (0.2, 0.25,
// 0.45), the velocity (u, v, w) is zeroed and P is set to Pinit * (0.5 - x).
// All output arrays are written in place, indexed by cell.
void initialisation_gaussian_3d(ArrayView<real_t, 1> ne, ArrayView<real_t, 1> u,
                                ArrayView<real_t, 1> v, ArrayView<real_t, 1> w,
                                ArrayView<real_t, 1> P,
                                ArrayView<const real_t, 2> cell_center,
                                real_t Pinit);

// Explicit forward-Euler cell-field update (translation of _update_new_value):
// ne_c(i) += dtime * ((rez_ne(i) + dissip_ne(i)) / cell_volume(i) + src_ne(i)).
// Common to all solvers, so it lives here rather than in any one solver. ne_c
// is updated in place. Has a GPU counterpart (launch_update_new_value).
void update_new_value(ArrayView<real_t, 1> ne_c,
                      ArrayView<const real_t, 1> rez_ne,
                      ArrayView<const real_t, 1> dissip_ne,
                      ArrayView<const real_t, 1> src_ne, real_t dtime,
                      ArrayView<const real_t, 1> cell_volume);
