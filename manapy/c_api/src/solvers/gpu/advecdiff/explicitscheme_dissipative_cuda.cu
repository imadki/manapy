#include <algorithm>

#include "advecdiff_compute.cuh"
#include "common/advecdiff/dissipative_common.hpp"

namespace {

// One thread per face; a grid-stride loop covers meshes larger than one grid.
// Scatter into dissip_w uses atomicAdd on the device (several faces touch the
// same cell), via the shared per-face routine.
__global__ void dissipative_kernel(
    ArrayView<const real_t, 1> wx_face, ArrayView<const real_t, 1> wy_face,
    ArrayView<const real_t, 1> wz_face,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_name, ArrayView<real_t, 1> dissip_w,
    real_t Dxx, real_t Dyy, real_t Dzz, index_t nbface) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbface; i += stride) {
    dissipative_face(i, wx_face, wy_face, wz_face, face_cellid, face_normal,
                     face_name, dissip_w, Dxx, Dyy, Dzz);
  }
}

__global__ void zero_kernel(ArrayView<real_t, 1> dissip_w, index_t nbcell) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t c =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       c < nbcell; c += stride) {
    dissip_w(c) = real_t(0);
  }
}

int grid_blocks(index_t n, int threads) {
  return static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
}

} // namespace

void launch_explicitscheme_dissipative(
    ArrayView<const real_t, 1> wx_face, ArrayView<const real_t, 1> wy_face,
    ArrayView<const real_t, 1> wz_face,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_name, ArrayView<real_t, 1> dissip_w,
    real_t Dxx, real_t Dyy, real_t Dzz, cudaStream_t stream) {
  const index_t nbcell = static_cast<index_t>(dissip_w.size(0));
  if (nbcell <= 0)
    return;

  constexpr int threads = 256;
  zero_kernel<<<grid_blocks(nbcell, threads), threads, 0, stream>>>(dissip_w,
                                                                    nbcell);

  const index_t nbface = static_cast<index_t>(face_cellid.size(0));
  if (nbface <= 0)
    return;
  dissipative_kernel<<<grid_blocks(nbface, threads), threads, 0, stream>>>(
      wx_face, wy_face, wz_face, face_cellid, face_normal, face_name, dissip_w,
      Dxx, Dyy, Dzz, nbface);
}
