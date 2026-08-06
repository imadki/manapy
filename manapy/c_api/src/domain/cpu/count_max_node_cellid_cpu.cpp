#include "domain_compute.hpp"


/*
  Determine the max neighboring cells of a node across all cells
*/
void count_max_node_cellid(ArrayView<const index_t, 2> cells,
                            ArrayView<index_t, 1> res) {
  const index_t nb_cells = static_cast<index_t>(cells.size(0));
  const index_t last = cells.size(1) - 1;

  for (index_t i = 0; i < nb_cells; ++i) {
    const index_t nb_nodes = cells(i, last);
    for (index_t k = 0; k < nb_nodes; ++k) {
      const index_t node = cells(i, k);
      res(node) += 1;
    }
  }
}
