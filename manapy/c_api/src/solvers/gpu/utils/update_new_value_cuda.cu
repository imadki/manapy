#include <algorithm>

#include "utils_compute.cuh"
#include "common/utils/update_new_value_common.hpp"

namespace {

// One thread per cell; a grid-stride loop covers meshes larger than one grid.
// Each thread reuses the shared host/device per-cell routine; ne_c is updated
// in place on the device.
__global__ void update_new_value_kernel(
    ArrayView<real_t, 1> ne_c, ArrayView<const real_t, 1> rez_ne,
    ArrayView<const real_t, 1> dissip_ne, ArrayView<const real_t, 1> src_ne,
    real_t dtime, ArrayView<const real_t, 1> cell_volume, index_t nbelements) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbelements; i += stride) {
    update_new_value_cell(i, ne_c, rez_ne, dissip_ne, src_ne, dtime,
                          cell_volume);
  }
}

} // namespace

void launch_update_new_value(ArrayView<real_t, 1> ne_c,
                             ArrayView<const real_t, 1> rez_ne,
                             ArrayView<const real_t, 1> dissip_ne,
                             ArrayView<const real_t, 1> src_ne, real_t dtime,
                             ArrayView<const real_t, 1> cell_volume,
                             cudaStream_t stream) {
  const index_t nbelements = static_cast<index_t>(ne_c.size(0));
  if (nbelements <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbelements) + threads - 1) / threads, 65535));
  update_new_value_kernel<<<blocks, threads, 0, stream>>>(
      ne_c, rez_ne, dissip_ne, src_ne, dtime, cell_volume, nbelements);
}
