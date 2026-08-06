#pragma once

#include "array_view.hpp"
#include "common/boundary_math.hpp"
#include "precision.hpp"

// Free-slip (slip wall) boundary condition applied to every halo ghost attached
// to halo node i whose boundary tag matches BCindex, in 3D (translation of
// haloghost_value_slip_3d in to_convert.py). Same reflection as the 2D version
// with the z component added; the face normal comes from columns 7-9 of
// ghost_ext_info_flt and is normalised here.
//
// The last column of node_haloghostid holds the number of valid entries in that
// row. Shared verbatim by the CPU loop (cpu/haloghost_value_slip_3d_cpu.cpp)
// and the CUDA kernel (gpu/haloghost_value_slip_3d_cuda.cu). Writes
// u_haloghost/v_haloghost/w_haloghost in place.
MANAPY_COMPUTE_HOST_DEVICE
void haloghost_value_slip_3d_node(
    index_t i, ArrayView<const real_t, 1> u_halo,
    ArrayView<const real_t, 1> v_halo, ArrayView<const real_t, 1> w_halo,
    ArrayView<real_t, 1> u_haloghost, ArrayView<real_t, 1> v_haloghost,
    ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex) {
  const index_t count = node_haloghostid(i, node_haloghostid.size(1) - 1);
  for (index_t j = 0; j < count; ++j) {
    const index_t ghost_id = node_haloghostid(i, j);
    if (ghost_ext_info_int(ghost_id, 1) != BCindex)
      continue;

    const index_t c = ghost_ext_info_int(ghost_id, 0);

    real_t nx = ghost_ext_info_flt(ghost_id, 7);
    real_t ny = ghost_ext_info_flt(ghost_id, 8);
    real_t nz = ghost_ext_info_flt(ghost_id, 9);
    const real_t nrm = boundary_sqrt(nx * nx + ny * ny + nz * nz);
    nx = nx / nrm;
    ny = ny / nrm;
    nz = nz / nrm;

    const real_t udotn = u_halo(c) * nx + v_halo(c) * ny + w_halo(c) * nz;
    u_haloghost(ghost_id) = u_halo(c) - real_t(2) * udotn * nx;
    v_haloghost(ghost_id) = v_halo(c) - real_t(2) * udotn * ny;
    w_haloghost(ghost_id) = w_halo(c) - real_t(2) * udotn * nz;
  }
}
