#include <algorithm>

#include "common/face_gradient_2d_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per entry in face_list; a grid-stride loop covers lists larger
// than one grid. ArrayViews are passed by value (their data pointers already
// reference device memory) and each thread reuses the shared host/device
// per-face routine, templated on which array supplies the far-side value.
template <FaceGradientKind Kind>
__global__ void face_gradient_2d_kernel(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face,
    ArrayView<const index_t, 1> face_list, index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = face_list(k);
    face_gradient_2d_face<Kind>(i, w_c, w_ghost, w_halo, w_node, face_cellid,
                                 faces, face_halofid, face_airDiamond, face_f1,
                                 face_f2, face_f3, face_f4, wx_face, wy_face);
  }
}

template <FaceGradientKind Kind>
void launch_face_gradient_2d_group(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face,
    ArrayView<const index_t, 1> face_list, cudaStream_t stream) {
  const index_t n = static_cast<index_t>(face_list.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  face_gradient_2d_kernel<Kind><<<blocks, threads, 0, stream>>>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      face_list, n);
}

} // namespace

void launch_face_gradient_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> dirichletfaces,
    ArrayView<const index_t, 1> neumann,
    ArrayView<const index_t, 1> d_periodicfaces, cudaStream_t stream) {
  launch_face_gradient_2d_group<FaceGradientKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_innerfaces, stream);
  launch_face_gradient_2d_group<FaceGradientKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_periodicfaces, stream);
  launch_face_gradient_2d_group<FaceGradientKind::Halo>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_halofaces, stream);
  launch_face_gradient_2d_group<FaceGradientKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      dirichletfaces, stream);
  launch_face_gradient_2d_group<FaceGradientKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      neumann, stream);
}
