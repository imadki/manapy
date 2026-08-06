#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Per-cell least-squares gradient for a single element i. Shared verbatim by
// the CPU loop (cpu/cell_gradient_2d_cpu.cpp) and the CUDA kernel
// (gpu/cell_gradient_2d_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE compiles it as a plain
// inline function in C++ TUs and as a __host__ __device__ function under nvcc,
// so the exact same arithmetic runs on both host and GPU. Writes w_x/w_y/w_z
// at index i in place.
MANAPY_COMPUTE_HOST_DEVICE
void cell_gradient_2d_element(
    index_t i, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const index_t, 2> cell_cellnid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> cell_ghostnid,
    ArrayView<const index_t, 2> cell_haloghostnid,
    ArrayView<const index_t, 2> cell_halonid,
    ArrayView<const index_t, 2> cells,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_x,
    ArrayView<real_t, 1> w_y, ArrayView<real_t, 1> w_z,
    ArrayView<const index_t, 1> ghost_faceid) {
  real_t i_xx = real_t(0);
  real_t i_yy = real_t(0);
  real_t i_xy = real_t(0);
  real_t j_xw = real_t(0);
  real_t j_yw = real_t(0);

  const real_t cx = cell_center(i, 0);
  const real_t cy = cell_center(i, 1);
  const real_t wi = w_c(i);

  {
    const index_t count = cell_cellnid(i, cell_cellnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = cell_cellnid(i, j);
      const real_t j_x = cell_center(cell, 0) - cx;
      const real_t j_y = cell_center(cell, 1) - cy;
      i_xx += j_x * j_x;
      i_yy += j_y * j_y;
      i_xy += j_x * j_y;
      j_xw += j_x * (w_c(cell) - wi);
      j_yw += j_y * (w_c(cell) - wi);
    }
  }

  {
    const index_t count = cell_ghostnid(i, cell_ghostnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = cell_ghostnid(i, j);
      const real_t j_x = ghost_info_flt(ghost_id, 0) - cx;
      const real_t j_y = ghost_info_flt(ghost_id, 1) - cy;
      i_xx += j_x * j_x;
      i_yy += j_y * j_y;
      i_xy += j_x * j_y;

      const real_t dw = w_ghost(ghost_faceid(ghost_id)) - wi;
      j_xw += j_x * dw;
      j_yw += j_y * dw;
    }
  }

  {
    const index_t nnod = cells(i, cells.size(1) - 1);
    for (index_t k = 0; k < nnod; ++k) {
      const index_t nod = cells(i, k);
      const index_t name = node_oldname(nod);

      if (name == index_t(11) || name == index_t(22)) {
        const index_t count =
            node_periodicid(nod, node_periodicid.size(1) - 1);
        for (index_t j = 0; j < count; ++j) {
          const index_t cell = node_periodicid(nod, j);
          const real_t j_x = cell_center(cell, 0) + cell_shift(cell, 0) - cx;
          const real_t j_y = cell_center(cell, 1) - cy;

          i_xx += j_x * j_x;
          i_yy += j_y * j_y;
          i_xy += j_x * j_y;
          j_xw += j_x * (w_c(cell) - wi);
          j_yw += j_y * (w_c(cell) - wi);
        }
      }

      if (name == index_t(33) || name == index_t(44)) {
        const index_t count =
            node_periodicid(nod, node_periodicid.size(1) - 1);
        for (index_t j = 0; j < count; ++j) {
          const index_t cell = node_periodicid(nod, j);
          const real_t j_x = cell_center(cell, 0) - cx;
          const real_t j_y = cell_center(cell, 1) + cell_shift(cell, 1) - cy;

          i_xx += j_x * j_x;
          i_yy += j_y * j_y;
          i_xy += j_x * j_y;
          j_xw += j_x * (w_c(cell) - wi);
          j_yw += j_y * (w_c(cell) - wi);
        }
      }
    }
  }

  {
    const index_t count = cell_halonid(i, cell_halonid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = cell_halonid(i, j);
      const real_t j_x = halo_centvol(cell, 0) - cx;
      const real_t j_y = halo_centvol(cell, 1) - cy;

      i_xx += j_x * j_x;
      i_yy += j_y * j_y;
      i_xy += j_x * j_y;
      j_xw += j_x * (w_halo(cell) - wi);
      j_yw += j_y * (w_halo(cell) - wi);
    }
  }

  {
    const index_t count = cell_haloghostnid(i, cell_haloghostnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = cell_haloghostnid(i, j);
      const real_t j_x = ghost_ext_info_flt(ghost_id, 0) - cx;
      const real_t j_y = ghost_ext_info_flt(ghost_id, 1) - cy;

      i_xx += j_x * j_x;
      i_yy += j_y * j_y;
      i_xy += j_x * j_y;
      j_xw += j_x * (w_haloghost(ghost_id) - wi);
      j_yw += j_y * (w_haloghost(ghost_id) - wi);
    }
  }

  const real_t dia = i_xx * i_yy - i_xy * i_xy;

  w_x(i) = (i_yy * j_xw - i_xy * j_yw) / dia;
  w_y(i) = (i_xx * j_yw - i_xy * j_xw) / dia;
  w_z(i) = real_t(0);
}
