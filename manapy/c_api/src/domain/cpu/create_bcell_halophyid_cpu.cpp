#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void create_bcell_halophyid(ArrayView<const index_t, 2> cells,
                             ArrayView<const index_t, 1> b_ncellid,
                             ArrayView<const index_t, 2> node_halophyid,
                             ArrayView<index_t, 1> i_visited,
                             ArrayView<index_t, 2> bcell_halophyid) {
  const index_t nb = static_cast<index_t>(b_ncellid.size(0));
  const index_t out_last = bcell_halophyid.size(1) - 1;

  for (index_t i = 0; i < nb; ++i) {
    const index_t bc = b_ncellid(i);
    const auto cell = cells.row(bc);

    bcell_halophyid(i, 0) = bc;
    index_t counter = 1;

    const index_t nb_nodes = cell(cell.size(0) - 1);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const auto node_nbf = node_halophyid.row(cell(j));
      const index_t count = node_nbf(node_nbf.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t v = node_nbf(k);
        if (i_visited(v) != i) {
          i_visited(v) = i;
          bcell_halophyid(i, counter) = v;
          ++counter;
        }
      }
    }
    bcell_halophyid(i, out_last) = counter;
  }
}
