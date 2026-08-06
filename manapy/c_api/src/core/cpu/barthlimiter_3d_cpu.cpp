#include "variable_compute.hpp"

#include "common/barthlimiter_3d_common.hpp"

void barthlimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center) {

  const index_t nbelement = static_cast<index_t>(w_c.size(0));

  for (index_t i = 0; i < nbelement; ++i) {
    barthlimiter_3d_cell(i, w_c, w_ghost, w_halo, w_x, w_y, w_z, face_cellid,
                          cell_faceid, face_name, face_haloid, cell_center,
                          face_center, psi);
  }
}
