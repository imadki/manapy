#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Explicit forward-Euler update of a cell field (translation of
// _update_new_value): advance ne_c(i) by dtime * ((rez + dissip) / volume +
// src). Common to all solvers (advec, advecdiff, ...), so it lives in
// solvers.utils rather than in any one solver. Per-cell and embarrassingly
// parallel; shared verbatim by the CPU loop and the CUDA kernel. Writes ne_c at
// index i in place.
MANAPY_COMPUTE_HOST_DEVICE
void update_new_value_cell(index_t i, ArrayView<real_t, 1> ne_c,
                           ArrayView<const real_t, 1> rez_ne,
                           ArrayView<const real_t, 1> dissip_ne,
                           ArrayView<const real_t, 1> src_ne, real_t dtime,
                           ArrayView<const real_t, 1> cell_volume) {
  ne_c(i) += dtime * ((rez_ne(i) + dissip_ne(i)) / cell_volume(i) + src_ne(i));
}
