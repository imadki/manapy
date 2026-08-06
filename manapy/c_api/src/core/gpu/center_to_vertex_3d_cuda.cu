#include <algorithm>

#include "common/center_to_vertex_3d_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per node; a grid-stride loop covers meshes larger than one grid.
// ArrayViews are passed by value (their data pointers already reference device
// memory) and each thread reuses the shared host/device node routine.
__global__ void center_to_vertex_3d_kernel(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
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
    ArrayView<const real_t, 1> node_R_z, ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const real_t, 1> node_lambda_z,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid, index_t nbnode) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbnode; i += stride) {
    center_to_vertex_3d_node(
        i, w_c, w_ghost, w_halo, w_haloghost, cell_center, halo_centvol,
        node_cellid, ghost_info_flt, ghost_ext_info_flt, node_ghostid,
        node_haloghostid, node_periodicid, node_halonid, nodes, node_oldname,
        node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y,
        node_lambda_z, node_number, cell_shift, w_n, ghost_faceid);
  }
}

} // namespace

void launch_center_to_vertex_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
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
    ArrayView<const real_t, 1> node_R_z, ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const real_t, 1> node_lambda_z,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream) {
  const index_t nbnode = static_cast<index_t>(nodes.size(0));
  if (nbnode <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbnode) + threads - 1) / threads, 65535));
  center_to_vertex_3d_kernel<<<blocks, threads, 0, stream>>>(
      w_c, w_ghost, w_halo, w_haloghost, cell_center, halo_centvol, node_cellid,
      ghost_info_flt, ghost_ext_info_flt, node_ghostid, node_haloghostid,
      node_periodicid, node_halonid, nodes, node_oldname, node_R_x, node_R_y,
      node_R_z, node_lambda_x, node_lambda_y, node_lambda_z, node_number,
      cell_shift, w_n, ghost_faceid, nbnode);
}
