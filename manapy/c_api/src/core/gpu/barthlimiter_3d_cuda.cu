#include <algorithm>

#include "common/barthlimiter_3d_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per cell; a grid-stride loop covers meshes larger than one grid.
// ArrayViews are passed by value (their data pointers already reference device
// memory) and each thread reuses the shared host/device per-cell routine.
__global__ void barthlimiter_3d_kernel(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center, index_t nbelement) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbelement; i += stride) {
    barthlimiter_3d_cell(i, w_c, w_ghost, w_halo, w_x, w_y, w_z, face_cellid,
                          cell_faceid, face_name, face_haloid, cell_center,
                          face_center, psi);
  }
}

} // namespace

void launch_barthlimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center, cudaStream_t stream) {
  const index_t nbelement = static_cast<index_t>(w_c.size(0));
  if (nbelement <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbelement) + threads - 1) / threads, 65535));
  barthlimiter_3d_kernel<<<blocks, threads, 0, stream>>>(
      w_c, w_ghost, w_halo, w_x, w_y, w_z, psi, face_cellid, cell_faceid,
      face_name, face_haloid, cell_center, face_center, nbelement);
}
