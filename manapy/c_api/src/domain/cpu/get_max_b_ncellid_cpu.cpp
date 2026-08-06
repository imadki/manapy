#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

index_t get_max_b_ncellid(ArrayView<const index_t, 1> b_nodeid,
                           ArrayView<const index_t, 2> node_cellid,
                           ArrayView<std::int8_t, 1> b_visited) {
  index_t cmp = 0;
  const index_t nb = static_cast<index_t>(b_nodeid.size(0));

  for (index_t i = 0; i < nb; ++i) {
    const index_t nodeid = b_nodeid(i);
    const auto neighbors = node_cellid.row(nodeid);
    const index_t count = neighbors(neighbors.size(0) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t cell = neighbors(j);
      if (b_visited(cell) == 0) {
        b_visited(cell) = 1;
        ++cmp;
      }
    }
  }
  return cmp;
}
