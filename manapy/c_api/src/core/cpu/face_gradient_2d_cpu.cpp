#include "variable_compute.hpp"

#include "common/face_gradient_2d_common.hpp"

namespace {

template <FaceGradientKind Kind>
void face_gradient_2d_group(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face,
    ArrayView<const index_t, 1> face_list) {
  const index_t n = static_cast<index_t>(face_list.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = face_list(k);
    face_gradient_2d_face<Kind>(i, w_c, w_ghost, w_halo, w_node, face_cellid,
                                 faces, face_halofid, face_airDiamond, face_f1,
                                 face_f2, face_f3, face_f4, wx_face, wy_face);
  }
}

} // namespace

void face_gradient_2d(
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
    ArrayView<const index_t, 1> d_periodicfaces) {
  face_gradient_2d_group<FaceGradientKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_innerfaces);
  face_gradient_2d_group<FaceGradientKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_periodicfaces);
  face_gradient_2d_group<FaceGradientKind::Halo>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      d_halofaces);
  face_gradient_2d_group<FaceGradientKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      dirichletfaces);
  face_gradient_2d_group<FaceGradientKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_halofid,
      face_airDiamond, face_f1, face_f2, face_f3, face_f4, wx_face, wy_face,
      neumann);
}
