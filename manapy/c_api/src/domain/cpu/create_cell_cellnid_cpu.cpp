#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

/*
Get all neighboring cells by collecting adjacent cells from each node of the cell.
*/
void create_cell_cellnid(ArrayView<const index_t, 2> cells,
                          ArrayView<const index_t, 2> node_cellid,
                          ArrayView<index_t, 2> cell_cellnid) {
  const index_t nb_cells = static_cast<index_t>(cells.size(0));
  const index_t cells_last = cells.size(1) - 1;
  const index_t cellnid_last = cell_cellnid.size(1) - 1;

  for (index_t i = 0; i < nb_cells; ++i) {
    const index_t nb_nodes = cells(i, cells_last);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const auto node_n = node_cellid.row(cells(i, j));
      const index_t count = node_n(node_n.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t nc = node_n(k);
        const index_t size = cell_cellnid(nc, cellnid_last);
        if (nc != i && (size == 0 || cell_cellnid(nc, size - 1) != i)) {
          cell_cellnid(nc, size) = i;
          cell_cellnid(nc, cellnid_last) += 1;
        }
      }
    }
  }
}
