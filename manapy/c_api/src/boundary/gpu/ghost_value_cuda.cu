#include <algorithm>

#include "boundary_compute.cuh"
#include "common/ghost_value_common.hpp"

namespace {

// One thread per entry in bc_faces; a grid-stride loop covers lists larger than
// one grid. ArrayViews are passed by value (their data pointers already
// reference device memory) and each thread reuses the shared host/device
// per-face routine, templated on which scalar condition to apply.
template <GhostValueKind Kind>
__global__ void ghost_value_kernel(ArrayView<const real_t, 1> w,
                                   ArrayView<real_t, 1> w_ghost,
                                   ArrayView<const index_t, 2> face_cellid,
                                   ArrayView<const index_t, 1> bc_faces,
                                   ArrayView<const real_t, 1> cst,
                                   ArrayView<const real_t, 1> face_dist_ortho,
                                   index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = bc_faces(k);
    ghost_value_face<Kind>(i, w, w_ghost, face_cellid, cst, face_dist_ortho);
  }
}

template <GhostValueKind Kind>
void launch_ghost_value_group(ArrayView<const real_t, 1> w,
                              ArrayView<real_t, 1> w_ghost,
                              ArrayView<const index_t, 2> face_cellid,
                              ArrayView<const index_t, 1> bc_faces,
                              ArrayView<const real_t, 1> cst,
                              ArrayView<const real_t, 1> face_dist_ortho,
                              cudaStream_t stream) {
  const index_t n = static_cast<index_t>(bc_faces.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  ghost_value_kernel<Kind><<<blocks, threads, 0, stream>>>(
      w, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho, n);
}

} // namespace

void launch_ghost_value_dirichlet(ArrayView<const real_t, 1> value,
                                  ArrayView<real_t, 1> w_ghost,
                                  ArrayView<const index_t, 2> face_cellid,
                                  ArrayView<const index_t, 1> bc_faces,
                                  ArrayView<const real_t, 1> cst,
                                  ArrayView<const real_t, 1> face_dist_ortho,
                                  cudaStream_t stream) {
  launch_ghost_value_group<GhostValueKind::Dirichlet>(
      value, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho, stream);
}

void launch_ghost_value_neumann(ArrayView<const real_t, 1> w_c,
                                ArrayView<real_t, 1> w_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 1> cst,
                                ArrayView<const real_t, 1> face_dist_ortho,
                                cudaStream_t stream) {
  launch_ghost_value_group<GhostValueKind::Neumann>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho, stream);
}

void launch_ghost_value_neumannNH(ArrayView<const real_t, 1> w_c,
                                  ArrayView<real_t, 1> w_ghost,
                                  ArrayView<const index_t, 2> face_cellid,
                                  ArrayView<const index_t, 1> bc_faces,
                                  ArrayView<const real_t, 1> cst,
                                  ArrayView<const real_t, 1> face_dist_ortho,
                                  cudaStream_t stream) {
  launch_ghost_value_group<GhostValueKind::NeumannNH>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho, stream);
}

void launch_ghost_value_nonslip(ArrayView<const real_t, 1> w_c,
                                ArrayView<real_t, 1> w_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 1> cst,
                                ArrayView<const real_t, 1> face_dist_ortho,
                                cudaStream_t stream) {
  launch_ghost_value_group<GhostValueKind::NonSlip>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho, stream);
}
