#include <algorithm>

#include "boundary_compute.cuh"
#include "common/haloghost_value_slip_3d_common.hpp"

namespace {

// One thread per entry in d_halonodes (each thread walks that node's halo
// ghosts); a grid-stride loop covers lists larger than one grid. Distinct nodes
// can list the same ghost id, but every matching thread writes the same value,
// so the races are benign -- as in the Python original.
__global__ void haloghost_value_slip_3d_kernel(
    ArrayView<const real_t, 1> u_halo, ArrayView<const real_t, 1> v_halo,
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> u_haloghost,
    ArrayView<real_t, 1> v_haloghost, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = d_halonodes(k);
    haloghost_value_slip_3d_node(i, u_halo, v_halo, w_halo, u_haloghost,
                                 v_haloghost, w_haloghost, node_haloghostid,
                                 ghost_ext_info_int, ghost_ext_info_flt,
                                 BCindex);
  }
}

} // namespace

void launch_haloghost_value_slip_3d(
    ArrayView<const real_t, 1> u_halo, ArrayView<const real_t, 1> v_halo,
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> u_haloghost,
    ArrayView<real_t, 1> v_haloghost, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, cudaStream_t stream) {
  const index_t n = static_cast<index_t>(d_halonodes.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  haloghost_value_slip_3d_kernel<<<blocks, threads, 0, stream>>>(
      u_halo, v_halo, w_halo, u_haloghost, v_haloghost, w_haloghost,
      node_haloghostid, ghost_ext_info_int, ghost_ext_info_flt, BCindex,
      d_halonodes, n);
}
