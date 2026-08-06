#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Distance-weighted center-to-vertex interpolation for a single node i. Shared
// verbatim by the CPU loop (cpu/center_to_vertex_2d_cpu.cpp) and the CUDA
// kernel (gpu/center_to_vertex_2d_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE compiles
// it as a plain inline function in C++ TUs and as a __host__ __device__
// function under nvcc, so the exact same arithmetic runs on both host and GPU.
// Writes w_n at index i in place.
//
// The linear-reconstruction weight is
//   alpha = (1 + lambda_x*dx + lambda_y*dy) / (n + lambda_x*R_x + lambda_y*R_y)
// with (dx, dy) the offset from node i to each contributing centre. The
// denominator is independent of the contribution, so it is evaluated once.
MANAPY_COMPUTE_HOST_DEVICE
void center_to_vertex_2d_node(
    index_t i, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> node_cellid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> node_ghostid,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 2> node_halonid, ArrayView<const real_t, 2> nodes,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 1> node_R_x, ArrayView<const real_t, 1> node_R_y,
    ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid) {
  const real_t nx = nodes(i, 0);
  const real_t ny = nodes(i, 1);
  const real_t lx = node_lambda_x(i);
  const real_t ly = node_lambda_y(i);
  const real_t denom = static_cast<real_t>(node_number(i)) +
                       lx * node_R_x(i) + ly * node_R_y(i);

  real_t acc = real_t(0);

  {
    const index_t count = node_cellid(i, node_cellid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = node_cellid(i, j);
      const real_t xdiff = cell_center(cell, 0) - nx;
      const real_t ydiff = cell_center(cell, 1) - ny;
      const real_t alpha = (real_t(1) + lx * xdiff + ly * ydiff) / denom;
      acc += alpha * w_c(cell);
    }
  }

  {
    const index_t count = node_ghostid(i, node_ghostid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = node_ghostid(i, j);
      const real_t xdiff = ghost_info_flt(ghost_id, 0) - nx;
      const real_t ydiff = ghost_info_flt(ghost_id, 1) - ny;
      const real_t alpha = (real_t(1) + lx * xdiff + ly * ydiff) / denom;
      acc += alpha * w_ghost(ghost_faceid(ghost_id));
    }
  }

  {
    const index_t count = node_haloghostid(i, node_haloghostid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = node_haloghostid(i, j);
      const real_t xdiff = ghost_ext_info_flt(ghost_id, 0) - nx;
      const real_t ydiff = ghost_ext_info_flt(ghost_id, 1) - ny;
      const real_t alpha = (real_t(1) + lx * xdiff + ly * ydiff) / denom;
      acc += alpha * w_haloghost(ghost_id);
    }
  }

  {
    const index_t count = node_halonid(i, node_halonid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = node_halonid(i, j);
      const real_t xdiff = halo_centvol(cell, 0) - nx;
      const real_t ydiff = halo_centvol(cell, 1) - ny;
      const real_t alpha = (real_t(1) + lx * xdiff + ly * ydiff) / denom;
      acc += alpha * w_halo(cell);
    }
  }

  // Unified periodic branch, full cell_shift (handles corner/edge nodes
  // carrying partners from more than one periodic direction -- each
  // partner cell already holds its own correctly-signed shift, zero on the
  // components it isn't periodic in).
  if (node_oldname(i) >= index_t(11)) {
    const index_t count = node_periodicid(i, node_periodicid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = node_periodicid(i, j);
      const real_t xdiff = cell_center(cell, 0) + cell_shift(cell, 0) - nx;
      const real_t ydiff = cell_center(cell, 1) + cell_shift(cell, 1) - ny;
      const real_t alpha = (real_t(1) + lx * xdiff + ly * ydiff) / denom;
      acc += alpha * w_c(cell);
    }
  }

  w_n(i) = acc;
}
