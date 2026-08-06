#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

index_t
count_max_bcell_halophyid(ArrayView<const index_t, 2> cells,
                           ArrayView<const index_t, 1> b_ncellid,
                           ArrayView<const index_t, 2> node_halophyid,
                           ArrayView<index_t, 1> i_visited) {
  index_t max_counter = 0;
  const index_t nb = static_cast<index_t>(b_ncellid.size(0));

  for (index_t i = 0; i < nb; ++i) {
    const index_t bc = b_ncellid(i);
    const auto cell = cells.row(bc);
    index_t counter = 0;

    const index_t nb_nodes = cell(cell.size(0) - 1);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const auto node_nbf = node_halophyid.row(cell(j));
      const index_t count = node_nbf(node_nbf.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t v = node_nbf(k);
        if (i_visited(v) != i) {
          i_visited(v) = i;
          ++counter;
        }
      }
    }
    if (counter > max_counter)
      max_counter = counter;
  }
  return max_counter;
}
