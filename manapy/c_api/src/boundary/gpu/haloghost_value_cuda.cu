#include <algorithm>

#include "boundary_compute.cuh"
#include "common/haloghost_value_common.hpp"

namespace {

// One thread per entry in d_halonodes (each thread walks that node's halo
// ghosts); a grid-stride loop covers lists larger than one grid. ArrayViews are
// passed by value (their data pointers already reference device memory) and
// each thread reuses the shared host/device per-node routine, templated on
// which scalar condition to apply.
//
// Distinct nodes can list the same ghost id (a boundary face contributes its
// ghost to every one of its nodes), so several threads may write the same
// w_haloghost entry. Every condition writes a value determined solely by
// ghost_id -- including NeumannNH, whose cst is indexed per ghost -- so all
// those threads store the same bytes and the races are benign: the result is
// identical to the CPU loop's, whatever the scheduling.
template <HaloGhostValueKind Kind>
__global__ void haloghost_value_kernel(
    ArrayView<const real_t, 1> w, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = d_halonodes(k);
    haloghost_value_node<Kind>(i, w, w_haloghost, node_haloghostid,
                               ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                               cst);
  }
}

template <HaloGhostValueKind Kind>
void launch_haloghost_value_group(
    ArrayView<const real_t, 1> w, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream) {
  const index_t n = static_cast<index_t>(d_halonodes.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  haloghost_value_kernel<Kind><<<blocks, threads, 0, stream>>>(
      w, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst, n);
}

} // namespace

void launch_haloghost_value_dirichlet(
    ArrayView<const real_t, 1> value_haloghost,
    ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream) {
  launch_haloghost_value_group<HaloGhostValueKind::Dirichlet>(
      value_haloghost, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst, stream);
}

void launch_haloghost_value_neumann(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream) {
  launch_haloghost_value_group<HaloGhostValueKind::Neumann>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst, stream);
}

void launch_haloghost_value_neumannNH(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream) {
  launch_haloghost_value_group<HaloGhostValueKind::NeumannNH>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst, stream);
}

void launch_haloghost_value_nonslip(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream) {
  launch_haloghost_value_group<HaloGhostValueKind::NonSlip>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst, stream);
}
