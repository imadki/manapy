#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

// Create neighboring cells for each node
void create_node_cellid(ArrayView<const index_t, 2> cells,
                         ArrayView<index_t, 2> node_cellid) {
  const index_t nb_cells = static_cast<index_t>(cells.size(0));
  const index_t cells_last = cells.size(1) - 1;
  const index_t node_last = node_cellid.size(1) - 1;

  for (index_t i = 0; i < nb_cells; ++i) {
    const index_t nb_nodes = cells(i, cells_last);
    for (index_t j = 0; j < nb_nodes; ++j) {
      auto node = node_cellid.row(cells(i, j));
      const index_t size = node(node_last);
      node(node_last) += 1;
      node(size) = i;
    }
  }

  const index_t nb_nodes_total = static_cast<index_t>(node_cellid.size(0));
  for (index_t i = 0; i < nb_nodes_total; ++i) {
    auto node = node_cellid.row(i);
    insertion_sort(node, node(node_last));
  }
}
