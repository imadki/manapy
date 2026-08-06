#pragma once

#include "array_view.hpp"
#include "common/boundary_math.hpp"
#include "precision.hpp"

// Free-slip (slip wall) boundary condition applied to every halo ghost attached
// to halo node i whose boundary tag matches BCindex, in 2D (translation of
// haloghost_value_slip_2d in to_convert.py). Same reflection as
// ghost_value_slip_2d_face:
//     U_haloghost = U_halo - 2 (U_halo . n) n
// but reading the halo cell from ghost_ext_info_int(ghost_id, 0) and the face
// normal from columns 7-8 of ghost_ext_info_flt. The normal is normalised here.
//
// The last column of node_haloghostid holds the number of valid entries in that
// row. Shared verbatim by the CPU loop (cpu/haloghost_value_slip_2d_cpu.cpp)
// and the CUDA kernel (gpu/haloghost_value_slip_2d_cuda.cu). Writes
// u_haloghost/v_haloghost in place.
MANAPY_COMPUTE_HOST_DEVICE
void haloghost_value_slip_2d_node(
    index_t i, ArrayView<const real_t, 1> u_halo,
    ArrayView<const real_t, 1> v_halo, ArrayView<real_t, 1> u_haloghost,
    ArrayView<real_t, 1> v_haloghost,
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
    const real_t nrm = boundary_sqrt(nx * nx + ny * ny);
    nx = nx / nrm;
    ny = ny / nrm;

    const real_t udotn = u_halo(c) * nx + v_halo(c) * ny;
    u_haloghost(ghost_id) = u_halo(c) - real_t(2) * udotn * nx;
    v_haloghost(ghost_id) = v_halo(c) - real_t(2) * udotn * ny;
  }
}
