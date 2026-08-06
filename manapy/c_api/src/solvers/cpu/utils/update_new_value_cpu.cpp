#include "utils_compute.hpp"

#include "common/utils/update_new_value_common.hpp"

// CPU entry point: forward-Euler update of every cell, calling the shared
// per-cell routine. ne_c is updated in place.
void update_new_value(ArrayView<real_t, 1> ne_c,
                      ArrayView<const real_t, 1> rez_ne,
                      ArrayView<const real_t, 1> dissip_ne,
                      ArrayView<const real_t, 1> src_ne, real_t dtime,
                      ArrayView<const real_t, 1> cell_volume) {
  const index_t nbelements = static_cast<index_t>(ne_c.size(0));
  for (index_t i = 0; i < nbelements; ++i)
    update_new_value_cell(i, ne_c, rez_ne, dissip_ne, src_ne, dtime,
                          cell_volume);
}
