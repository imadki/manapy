#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Which array (and indexing scheme) supplies the value on the far side of a
// face for a single face-gradient entry (translation of _face_gradient_3d in
// to_convert.py, which repeats the same math across five index lists that
// differ only in how the far-side value is fetched):
//   TwoCell -> w_c(face_cellid(i, 1))   interior faces (d_innerfaces) and
//                                       periodic faces (d_periodicboundaryfaces)
//   Halo    -> w_halo(face_haloid(i))   halo faces (d_halofaces)
//   Ghost   -> w_ghost(i)               Dirichlet and Neumann boundary faces
//                                       (dirichletfaces, neumann)
enum class FaceGradient3DKind { TwoCell, Halo, Ghost };

// Per-face gradient contribution for a single face i on a 3D unstructured
// mesh. Shared verbatim by the CPU loops (cpu/face_gradient_3d_cpu.cpp) and
// the CUDA kernels (gpu/face_gradient_3d_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE
// compiles it as a plain inline function in C++ TUs and as a
// __host__ __device__ function under nvcc, so the exact same arithmetic runs
// on both host and GPU. Writes wx_face/wy_face/wz_face at index i in place.
//
// faces holds up to 4 node ids per row with the valid count in the last
// column (3 for a triangular face, 4 for a quad face); i_4 falls back to i_3
// on triangular faces, matching the Python original.
template <FaceGradient3DKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void face_gradient_3d_face(
    index_t i, ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 1> face_air_diamond,
    ArrayView<const real_t, 2> face_normal, ArrayView<const real_t, 2> face_f1,
    ArrayView<const real_t, 2> face_f2, ArrayView<real_t, 1> wx_face,
    ArrayView<real_t, 1> wy_face, ArrayView<real_t, 1> wz_face) {
  const real_t v_l = w_c(face_cellid(i, 0));

  real_t v_r;
  if constexpr (Kind == FaceGradient3DKind::TwoCell) {
    v_r = w_c(face_cellid(i, 1));
  } else if constexpr (Kind == FaceGradient3DKind::Halo) {
    v_r = w_halo(face_haloid(i));
  } else {
    v_r = w_ghost(i);
  }

  const index_t nnod = faces(i, faces.size(1) - 1);
  const index_t i1 = faces(i, 0);
  const index_t i2 = faces(i, 1);
  const index_t i3 = faces(i, 2);
  const index_t i4 = (nnod == index_t(4)) ? faces(i, 3) : i3;

  const real_t v_a = w_node(i1);
  const real_t v_b = w_node(i2);
  const real_t v_c = w_node(i3);
  const real_t v_d = w_node(i4);

  const real_t inv = real_t(1) / face_air_diamond(i);

  wx_face(i) = (face_f1(i, 0) * (v_a - v_c) + face_f2(i, 0) * (v_b - v_d) +
               face_normal(i, 0) * (v_r - v_l)) *
              inv;
  wy_face(i) = (face_f1(i, 1) * (v_a - v_c) + face_f2(i, 1) * (v_b - v_d) +
               face_normal(i, 1) * (v_r - v_l)) *
              inv;
  wz_face(i) = (face_f1(i, 2) * (v_a - v_c) + face_f2(i, 2) * (v_b - v_d) +
               face_normal(i, 2) * (v_r - v_l)) *
              inv;
}
