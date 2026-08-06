#include "boundary_compute.hpp"

#include "common/haloghost_value_common.hpp"

namespace {

// The four scalar conditions differ only in the Kind tag: same gather over
// d_halonodes, same arguments, one shared loop body. `w` is the prescribed
// per-halo-ghost value array for Dirichlet and the halo cell field for the
// others (see common/haloghost_value_common.hpp).
template <HaloGhostValueKind Kind>
void haloghost_value_group(ArrayView<const real_t, 1> w,
                           ArrayView<real_t, 1> w_haloghost,
                           ArrayView<const index_t, 2> node_haloghostid,
                           ArrayView<const index_t, 2> ghost_ext_info_int,
                           ArrayView<const real_t, 2> ghost_ext_info_flt,
                           index_t BCindex,
                           ArrayView<const index_t, 1> d_halonodes,
                           ArrayView<const real_t, 1> cst) {
  const index_t n = static_cast<index_t>(d_halonodes.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = d_halonodes(k);
    haloghost_value_node<Kind>(i, w, w_haloghost, node_haloghostid,
                               ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                               cst);
  }
}

} // namespace

void haloghost_value_dirichlet(ArrayView<const real_t, 1> value_haloghost,
                               ArrayView<real_t, 1> w_haloghost,
                               ArrayView<const index_t, 2> node_haloghostid,
                               ArrayView<const index_t, 2> ghost_ext_info_int,
                               ArrayView<const real_t, 2> ghost_ext_info_flt,
                               index_t BCindex,
                               ArrayView<const index_t, 1> d_halonodes,
                               ArrayView<const real_t, 1> cst) {
  haloghost_value_group<HaloGhostValueKind::Dirichlet>(
      value_haloghost, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst);
}

void haloghost_value_neumann(ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes,
                             ArrayView<const real_t, 1> cst) {
  haloghost_value_group<HaloGhostValueKind::Neumann>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst);
}

void haloghost_value_neumannNH(ArrayView<const real_t, 1> w_halo,
                               ArrayView<real_t, 1> w_haloghost,
                               ArrayView<const index_t, 2> node_haloghostid,
                               ArrayView<const index_t, 2> ghost_ext_info_int,
                               ArrayView<const real_t, 2> ghost_ext_info_flt,
                               index_t BCindex,
                               ArrayView<const index_t, 1> d_halonodes,
                               ArrayView<const real_t, 1> cst) {
  haloghost_value_group<HaloGhostValueKind::NeumannNH>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst);
}

void haloghost_value_nonslip(ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes,
                             ArrayView<const real_t, 1> cst) {
  haloghost_value_group<HaloGhostValueKind::NonSlip>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst);
}
