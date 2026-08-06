#include "variable_compute.hpp"

#include "common/center_to_vertex_3d_common.hpp"

void center_to_vertex_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> node_cellid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> node_ghostid,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 2> node_halonid, ArrayView<const real_t, 2> nodes,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 1> node_R_x, ArrayView<const real_t, 1> node_R_y,
    ArrayView<const real_t, 1> node_R_z, ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const real_t, 1> node_lambda_z,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid) {

  const index_t nbnode = static_cast<index_t>(nodes.size(0));

  for (index_t i = 0; i < nbnode; ++i) {
    center_to_vertex_3d_node(
        i, w_c, w_ghost, w_halo, w_haloghost, cell_center, halo_centvol,
        node_cellid, ghost_info_flt, ghost_ext_info_flt, node_ghostid,
        node_haloghostid, node_periodicid, node_halonid, nodes, node_oldname,
        node_R_x, node_R_y, node_R_z, node_lambda_x, node_lambda_y,
        node_lambda_z, node_number, cell_shift, w_n, ghost_faceid);
  }
}
