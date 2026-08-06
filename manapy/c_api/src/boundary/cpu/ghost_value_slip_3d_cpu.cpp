#include "boundary_compute.hpp"

#include "common/ghost_value_slip_3d_common.hpp"

void ghost_value_slip_3d(ArrayView<const real_t, 1> u_c,
                         ArrayView<const real_t, 1> v_c,
                         ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> u_ghost,
                         ArrayView<real_t, 1> v_ghost,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 2> normal) {
  const index_t n = static_cast<index_t>(bc_faces.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = bc_faces(k);
    ghost_value_slip_3d_face(i, u_c, v_c, w_c, u_ghost, v_ghost, w_ghost,
                             face_cellid, normal);
  }
}
