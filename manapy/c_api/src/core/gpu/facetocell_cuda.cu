#include <algorithm>

#include "common/facetocell_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per cell; a grid-stride loop covers meshes larger than one grid.
// ArrayViews are passed by value (their data pointers already reference device
// memory) and each thread reuses the shared host/device per-cell routine.
__global__ void facetocell_kernel(ArrayView<const real_t, 1> u_face,
                                   ArrayView<const index_t, 2> cell_faceid,
                                   ArrayView<real_t, 1> u_c, index_t nbelement) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbelement; i += stride) {
    facetocell_cell(i, u_face, cell_faceid, u_c);
  }
}

} // namespace

void launch_facetocell(ArrayView<const real_t, 1> u_face,
                        ArrayView<const index_t, 2> cell_faceid,
                        ArrayView<real_t, 1> u_c, cudaStream_t stream) {
  const index_t nbelement = static_cast<index_t>(u_c.size(0));
  if (nbelement <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbelement) + threads - 1) / threads, 65535));
  facetocell_kernel<<<blocks, threads, 0, stream>>>(u_face, cell_faceid, u_c,
                                                      nbelement);
}
