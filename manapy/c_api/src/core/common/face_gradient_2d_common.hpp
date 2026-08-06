#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Which array (and indexing scheme) supplies the value on the far side of a
// face for a single face-gradient entry (translation of _face_gradient_2d in
// to_convert.py, which repeats the same math across five index lists that
// differ only in how the far-side value is fetched):
//   TwoCell -> w_c(face_cellid(i, 1))   interior faces (d_innerfaces) and
//                                       periodic faces (d_periodicfaces)
//   Halo    -> w_halo(face_halofid(i))  halo faces (d_halofaces)
//   Ghost   -> w_ghost(i)               Dirichlet and Neumann boundary faces
//                                       (dirichletfaces, neumann)
enum class FaceGradientKind { TwoCell, Halo, Ghost };

// Per-face Green-Gauss-style gradient contribution for a single face i.
// Shared verbatim by the CPU loops (cpu/face_gradient_2d_cpu.cpp) and the CUDA
// kernels (gpu/face_gradient_2d_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE compiles
// it as a plain inline function in C++ TUs and as a __host__ __device__
// function under nvcc, so the exact same arithmetic runs on both host and
// GPU. Writes wx_face/wy_face at index i in place.
template <FaceGradientKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void face_gradient_2d_face(
    index_t i, ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face) {
  const real_t vv1 = w_c(face_cellid(i, 0));

  real_t vv2;
  if constexpr (Kind == FaceGradientKind::TwoCell) {
    vv2 = w_c(face_cellid(i, 1));
  } else if constexpr (Kind == FaceGradientKind::Halo) {
    vv2 = w_halo(face_halofid(i));
  } else {
    vv2 = w_ghost(i);
  }

  const real_t vi1 = w_node(faces(i, 0));
  const real_t vi2 = w_node(faces(i, 1));

  const real_t inv2d = real_t(1) / (real_t(2) * face_airDiamond(i));

  wx_face(i) = inv2d * ((vi1 + vv1) * face_f1(i, 1) + (vv1 + vi2) * face_f2(i, 1) +
                        (vi2 + vv2) * face_f3(i, 1) + (vv2 + vi1) * face_f4(i, 1));
  wy_face(i) = -inv2d * ((vi1 + vv1) * face_f1(i, 0) + (vv1 + vi2) * face_f2(i, 0) +
                         (vi2 + vv2) * face_f3(i, 0) + (vv2 + vi1) * face_f4(i, 0));
}
