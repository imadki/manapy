#include "boundary_compute.hpp"

#include "common/ghost_value_common.hpp"

namespace {

// The four scalar conditions differ only in the Kind tag: same gather over
// bc_faces, same arguments, one shared loop body.
template <GhostValueKind Kind>
void ghost_value_group(ArrayView<const real_t, 1> w,
                       ArrayView<real_t, 1> w_ghost,
                       ArrayView<const index_t, 2> face_cellid,
                       ArrayView<const index_t, 1> bc_faces,
                       ArrayView<const real_t, 1> cst,
                       ArrayView<const real_t, 1> face_dist_ortho) {
  const index_t n = static_cast<index_t>(bc_faces.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = bc_faces(k);
    ghost_value_face<Kind>(i, w, w_ghost, face_cellid, cst, face_dist_ortho);
  }
}

} // namespace

void ghost_value_dirichlet(ArrayView<const real_t, 1> value,
                           ArrayView<real_t, 1> w_ghost,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const index_t, 1> bc_faces,
                           ArrayView<const real_t, 1> cst,
                           ArrayView<const real_t, 1> face_dist_ortho) {
  ghost_value_group<GhostValueKind::Dirichlet>(value, w_ghost, face_cellid,
                                               bc_faces, cst, face_dist_ortho);
}

void ghost_value_neumann(ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 1> cst,
                         ArrayView<const real_t, 1> face_dist_ortho) {
  ghost_value_group<GhostValueKind::Neumann>(w_c, w_ghost, face_cellid,
                                             bc_faces, cst, face_dist_ortho);
}

void ghost_value_neumannNH(ArrayView<const real_t, 1> w_c,
                           ArrayView<real_t, 1> w_ghost,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const index_t, 1> bc_faces,
                           ArrayView<const real_t, 1> cst,
                           ArrayView<const real_t, 1> face_dist_ortho) {
  ghost_value_group<GhostValueKind::NeumannNH>(w_c, w_ghost, face_cellid,
                                               bc_faces, cst, face_dist_ortho);
}

void ghost_value_nonslip(ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 1> cst,
                         ArrayView<const real_t, 1> face_dist_ortho) {
  ghost_value_group<GhostValueKind::NonSlip>(w_c, w_ghost, face_cellid,
                                             bc_faces, cst, face_dist_ortho);
}
