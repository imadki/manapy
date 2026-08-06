#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Per-cell least-squares gradient for a single element i on a 3D unstructured
// mesh. Shared verbatim by the CPU loop (cpu/cell_gradient_3d_cpu.cpp) and the
// CUDA kernel (gpu/cell_gradient_3d_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE
// compiles it as a plain inline function in C++ TUs and as a
// __host__ __device__ function under nvcc, so the exact same arithmetic runs
// on both host and GPU. Writes w_x/w_y/w_z at index i in place.
//
// Periodic neighbours are looked up directly through cell_periodicfid (unlike
// the 2D kernel, which resolves them via node_periodicid/node_oldname).
MANAPY_COMPUTE_HOST_DEVICE
void cell_gradient_3d_element(
    index_t i, ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const index_t, 2> cell_cellnid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> cell_ghostnid,
    ArrayView<const index_t, 2> cell_haloghostnid,
    ArrayView<const index_t, 2> cell_halonid,
    ArrayView<const index_t, 2> cell_periodicfid,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_x,
    ArrayView<real_t, 1> w_y, ArrayView<real_t, 1> w_z,
    ArrayView<const index_t, 1> ghost_faceid) {
  real_t i_xx = real_t(0);
  real_t i_yy = real_t(0);
  real_t i_zz = real_t(0);
  real_t i_xy = real_t(0);
  real_t i_xz = real_t(0);
  real_t i_yz = real_t(0);
  real_t j_x = real_t(0);
  real_t j_y = real_t(0);
  real_t j_z = real_t(0);

  const real_t cx = cell_center(i, 0);
  const real_t cy = cell_center(i, 1);
  const real_t cz = cell_center(i, 2);
  const real_t wi = w_c(i);

  {
    const index_t count = cell_cellnid(i, cell_cellnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = cell_cellnid(i, j);
      const real_t jx = cell_center(cell, 0) - cx;
      const real_t jy = cell_center(cell, 1) - cy;
      const real_t jz = cell_center(cell, 2) - cz;
      i_xx += jx * jx;
      i_yy += jy * jy;
      i_zz += jz * jz;
      i_xy += jx * jy;
      i_xz += jx * jz;
      i_yz += jy * jz;
      const real_t dw = w_c(cell) - wi;
      j_x += jx * dw;
      j_y += jy * dw;
      j_z += jz * dw;
    }
  }

  {
    const index_t count = cell_ghostnid(i, cell_ghostnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = cell_ghostnid(i, j);
      const real_t jx = ghost_info_flt(ghost_id, 0) - cx;
      const real_t jy = ghost_info_flt(ghost_id, 1) - cy;
      const real_t jz = ghost_info_flt(ghost_id, 2) - cz;
      i_xx += jx * jx;
      i_yy += jy * jy;
      i_zz += jz * jz;
      i_xy += jx * jy;
      i_xz += jx * jz;
      i_yz += jy * jz;
      const real_t dw = w_ghost(ghost_faceid(ghost_id)) - wi;
      j_x += jx * dw;
      j_y += jy * dw;
      j_z += jz * dw;
    }
  }

  {
    const index_t count = cell_periodicfid(i, cell_periodicfid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = cell_periodicfid(i, j);
      const real_t jx = cell_center(cell, 0) + cell_shift(cell, 0) - cx;
      const real_t jy = cell_center(cell, 1) + cell_shift(cell, 1) - cy;
      const real_t jz = cell_center(cell, 2) + cell_shift(cell, 2) - cz;
      i_xx += jx * jx;
      i_yy += jy * jy;
      i_zz += jz * jz;
      i_xy += jx * jy;
      i_xz += jx * jz;
      i_yz += jy * jz;
      const real_t dw = w_c(cell) - wi;
      j_x += jx * dw;
      j_y += jy * dw;
      j_z += jz * dw;
    }
  }

  {
    const index_t count = cell_halonid(i, cell_halonid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = cell_halonid(i, j);
      const real_t jx = halo_centvol(cell, 0) - cx;
      const real_t jy = halo_centvol(cell, 1) - cy;
      const real_t jz = halo_centvol(cell, 2) - cz;
      i_xx += jx * jx;
      i_yy += jy * jy;
      i_zz += jz * jz;
      i_xy += jx * jy;
      i_xz += jx * jz;
      i_yz += jy * jz;
      const real_t dw = w_halo(cell) - wi;
      j_x += jx * dw;
      j_y += jy * dw;
      j_z += jz * dw;
    }
  }

  {
    const index_t count = cell_haloghostnid(i, cell_haloghostnid.size(1) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t ghost_id = cell_haloghostnid(i, j);
      const real_t jx = ghost_ext_info_flt(ghost_id, 0) - cx;
      const real_t jy = ghost_ext_info_flt(ghost_id, 1) - cy;
      const real_t jz = ghost_ext_info_flt(ghost_id, 2) - cz;
      i_xx += jx * jx;
      i_yy += jy * jy;
      i_zz += jz * jz;
      i_xy += jx * jy;
      i_xz += jx * jz;
      i_yz += jy * jz;
      const real_t dw = w_haloghost(ghost_id) - wi;
      j_x += jx * dw;
      j_y += jy * dw;
      j_z += jz * dw;
    }
  }

  const real_t dia = i_xx * i_yy * i_zz + real_t(2) * i_xy * i_xz * i_yz -
                     i_xx * i_yz * i_yz - i_yy * i_xz * i_xz - i_zz * i_xy * i_xy;

  w_x(i) = ((i_yy * i_zz - i_yz * i_yz) * j_x + (i_xz * i_yz - i_xy * i_zz) * j_y +
            (i_xy * i_yz - i_xz * i_yy) * j_z) /
           dia;
  w_y(i) = ((i_xz * i_yz - i_xy * i_zz) * j_x + (i_xx * i_zz - i_xz * i_xz) * j_y +
            (i_xy * i_xz - i_yz * i_xx) * j_z) /
           dia;
  w_z(i) = ((i_xy * i_yz - i_xz * i_yy) * j_x + (i_xy * i_xz - i_yz * i_xx) * j_y +
            (i_xx * i_yy - i_xy * i_xy) * j_z) /
           dia;
}
