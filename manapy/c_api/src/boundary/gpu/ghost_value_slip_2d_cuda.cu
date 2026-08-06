#include <algorithm>

#include "boundary_compute.cuh"
#include "common/ghost_value_slip_2d_common.hpp"

namespace {

// One thread per entry in bc_faces; a grid-stride loop covers lists larger than
// one grid. ArrayViews are passed by value (their data pointers already
// reference device memory) and each thread reuses the shared host/device
// per-face routine.
__global__ void ghost_value_slip_2d_kernel(
    ArrayView<const real_t, 1> u_c, ArrayView<const real_t, 1> v_c,
    ArrayView<real_t, 1> u_ghost, ArrayView<real_t, 1> v_ghost,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 1> bc_faces, ArrayView<const real_t, 2> normal,
    index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = bc_faces(k);
    ghost_value_slip_2d_face(i, u_c, v_c, u_ghost, v_ghost, face_cellid, normal);
  }
}

} // namespace

void launch_ghost_value_slip_2d(ArrayView<const real_t, 1> u_c,
                                ArrayView<const real_t, 1> v_c,
                                ArrayView<real_t, 1> u_ghost,
                                ArrayView<real_t, 1> v_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 2> normal,
                                cudaStream_t stream) {
  const index_t n = static_cast<index_t>(bc_faces.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  ghost_value_slip_2d_kernel<<<blocks, threads, 0, stream>>>(
      u_c, v_c, u_ghost, v_ghost, face_cellid, bc_faces, normal, n);
}
