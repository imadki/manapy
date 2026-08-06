#include "variable_compute.hpp"

#include "common/face_gradient_3d_common.hpp"

namespace {

template <FaceGradient3DKind Kind>
void face_gradient_3d_group(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 1> face_air_diamond,
    ArrayView<const real_t, 2> face_normal, ArrayView<const real_t, 2> face_f1,
    ArrayView<const real_t, 2> face_f2, ArrayView<real_t, 1> wx_face,
    ArrayView<real_t, 1> wy_face, ArrayView<real_t, 1> wz_face,
    ArrayView<const index_t, 1> face_list) {
  const index_t n = static_cast<index_t>(face_list.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = face_list(k);
    face_gradient_3d_face<Kind>(i, w_c, w_ghost, w_halo, w_node, face_cellid,
                                 faces, face_haloid, face_air_diamond,
                                 face_normal, face_f1, face_f2, wx_face,
                                 wy_face, wz_face);
  }
}

} // namespace

void face_gradient_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 1> face_air_diamond,
    ArrayView<const real_t, 2> face_normal, ArrayView<const real_t, 2> face_f1,
    ArrayView<const real_t, 2> face_f2, ArrayView<real_t, 1> wx_face,
    ArrayView<real_t, 1> wy_face, ArrayView<real_t, 1> wz_face,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> dirichletfaces,
    ArrayView<const index_t, 1> neumann,
    ArrayView<const index_t, 1> d_periodicboundaryfaces) {
  face_gradient_3d_group<FaceGradient3DKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_haloid,
      face_air_diamond, face_normal, face_f1, face_f2, wx_face, wy_face,
      wz_face, d_innerfaces);
  face_gradient_3d_group<FaceGradient3DKind::TwoCell>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_haloid,
      face_air_diamond, face_normal, face_f1, face_f2, wx_face, wy_face,
      wz_face, d_periodicboundaryfaces);
  face_gradient_3d_group<FaceGradient3DKind::Halo>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_haloid,
      face_air_diamond, face_normal, face_f1, face_f2, wx_face, wy_face,
      wz_face, d_halofaces);
  face_gradient_3d_group<FaceGradient3DKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_haloid,
      face_air_diamond, face_normal, face_f1, face_f2, wx_face, wy_face,
      wz_face, dirichletfaces);
  face_gradient_3d_group<FaceGradient3DKind::Ghost>(
      w_c, w_ghost, w_halo, w_node, face_cellid, faces, face_haloid,
      face_air_diamond, face_normal, face_f1, face_f2, wx_face, wy_face,
      wz_face, neumann);
}
