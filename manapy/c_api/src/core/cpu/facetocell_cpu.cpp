#include "variable_compute.hpp"

#include "common/facetocell_common.hpp"

void facetocell(ArrayView<const real_t, 1> u_face,
                 ArrayView<const index_t, 2> cell_faceid, ArrayView<real_t, 1> u_c) {
  const index_t nbelement = static_cast<index_t>(u_c.size(0));

  for (index_t i = 0; i < nbelement; ++i) {
    facetocell_cell(i, u_face, cell_faceid, u_c);
  }
}
