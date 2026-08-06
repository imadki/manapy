#pragma once

#include "array_view.hpp"
#include "common/boundary_math.hpp"
#include "precision.hpp"

// Free-slip (slip wall) boundary condition for a single boundary face i in 3D
// (translation of ghost_value_slip_3d in to_convert.py). Same reflection as the
// 2D version, with the z component added:
//     U_ghost = U_c - 2 (U_c . n) n
// The normal is normalised here, so it works whether `normal` is unit or
// area-scaled.
//
// Shared verbatim by the CPU loop (cpu/ghost_value_slip_3d_cpu.cpp) and the
// CUDA kernel (gpu/ghost_value_slip_3d_cuda.cu). Writes u_ghost/v_ghost/w_ghost
// at index i in place.
MANAPY_COMPUTE_HOST_DEVICE
void ghost_value_slip_3d_face(index_t i, ArrayView<const real_t, 1> u_c,
                              ArrayView<const real_t, 1> v_c,
                              ArrayView<const real_t, 1> w_c,
                              ArrayView<real_t, 1> u_ghost,
                              ArrayView<real_t, 1> v_ghost,
                              ArrayView<real_t, 1> w_ghost,
                              ArrayView<const index_t, 2> face_cellid,
                              ArrayView<const real_t, 2> normal) {
  const index_t c = face_cellid(i, 0);

  real_t nx = normal(i, 0);
  real_t ny = normal(i, 1);
  real_t nz = normal(i, 2);
  const real_t nrm = boundary_sqrt(nx * nx + ny * ny + nz * nz);
  nx = nx / nrm;
  ny = ny / nrm;
  nz = nz / nrm;

  const real_t udotn = u_c(c) * nx + v_c(c) * ny + w_c(c) * nz;
  u_ghost(i) = u_c(c) - real_t(2) * udotn * nx;
  v_ghost(i) = v_c(c) - real_t(2) * udotn * ny;
  w_ghost(i) = w_c(c) - real_t(2) * udotn * nz;
}
