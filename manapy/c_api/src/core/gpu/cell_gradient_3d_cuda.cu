#include <algorithm>

#include "common/cell_gradient_3d_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per cell; a grid-stride loop covers meshes larger than one grid.
// ArrayViews are passed by value (their data pointers already reference device
// memory) and each thread reuses the shared host/device element routine.
__global__ void cell_gradient_3d_kernel(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
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
    ArrayView<const index_t, 1> ghost_faceid, index_t nbelement) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbelement; i += stride) {
    cell_gradient_3d_element(
        i, w_c, w_ghost, w_halo, w_haloghost, cell_center, cell_cellnid,
        ghost_info_flt, ghost_ext_info_flt, cell_ghostnid, cell_haloghostnid,
        cell_halonid, cell_periodicfid, halo_centvol, cell_shift, w_x, w_y,
        w_z, ghost_faceid);
  }
}

} // namespace

void launch_cell_gradient_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
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
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream) {
  const index_t nbelement = static_cast<index_t>(w_c.size(0));
  if (nbelement <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbelement) + threads - 1) / threads, 65535));
  cell_gradient_3d_kernel<<<blocks, threads, 0, stream>>>(
      w_c, w_ghost, w_halo, w_haloghost, cell_center, cell_cellnid,
      ghost_info_flt, ghost_ext_info_flt, cell_ghostnid, cell_haloghostnid,
      cell_halonid, cell_periodicfid, halo_centvol, cell_shift, w_x, w_y, w_z,
      ghost_faceid, nbelement);
}
