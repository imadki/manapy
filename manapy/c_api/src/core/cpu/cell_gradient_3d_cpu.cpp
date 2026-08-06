#include "variable_compute.hpp"

#include "common/cell_gradient_3d_common.hpp"

void cell_gradient_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const index_t, 2> cell_cellnid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> cell_ghostnid,
    ArrayView<const index_t, 2> cell_haloghostnid,
    ArrayView<const index_t, 2> cell_halonid,
    ArrayView<const index_t, 2> cell_periodicfid,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_x,
    ArrayView<real_t, 1> w_y, ArrayView<real_t, 1> w_z,
    ArrayView<const index_t, 1> ghost_faceid) {

  const index_t nbelement = static_cast<index_t>(w_c.size(0));

  for (index_t i = 0; i < nbelement; ++i) {
    cell_gradient_3d_element(
        i, w_c, w_ghost, w_halo, w_haloghost, cell_center, cell_cellnid,
        ghost_info_flt, ghost_ext_info_flt, cell_ghostnid, cell_haloghostnid,
        cell_halonid, cell_periodicfid, halo_centvol, cell_shift, w_x, w_y,
        w_z, ghost_faceid);
  }
}
