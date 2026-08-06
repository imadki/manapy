#include "boundary_compute.hpp"

#include "common/haloghost_value_slip_3d_common.hpp"

void haloghost_value_slip_3d(ArrayView<const real_t, 1> u_halo,
                             ArrayView<const real_t, 1> v_halo,
                             ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> u_haloghost,
                             ArrayView<real_t, 1> v_haloghost,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes) {
  const index_t n = static_cast<index_t>(d_halonodes.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = d_halonodes(k);
    haloghost_value_slip_3d_node(i, u_halo, v_halo, w_halo, u_haloghost,
                                 v_haloghost, w_haloghost, node_haloghostid,
                                 ghost_ext_info_int, ghost_ext_info_flt,
                                 BCindex);
  }
}
