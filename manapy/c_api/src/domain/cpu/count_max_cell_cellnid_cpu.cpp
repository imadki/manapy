#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"


/*
    Get the maximum number of neighboring cells per cell's nodes across the mesh

    Details:
    For each cell in the mesh, we need to examine its nodes and count the cells that neighbor those nodes.
    to get all neighboring cells of the cell
    Then, determine the highest number of neighboring cells

    Args:
      cells: (cell_id => nodes of the cell)
      node_cellid: (node_id => neighboring cells of the node)

    Return:
      Maximum number of neighboring cells per cell's nodes across the mesh

    Implementation details:
      to ensure that a neighboring cell is visited only once, we set `visited[neighbor_cell] = cell_id`
      thus for the same neighboring cell `visited[neighbor_cell]` is already set by `cell_id`
      for the next cell `visited` will automatically reset because next_cell_id != all_old_cell_id
*/
index_t count_max_cell_cellnid(ArrayView<const index_t, 2> cells,
                                ArrayView<const index_t, 2> node_cellid,
                                ArrayView<index_t, 1> i_visited) {
  const index_t nb_cells = static_cast<index_t>(cells.size(0));
  const index_t cells_last = cells.size(1) - 1;
  index_t max_counter = 0;

  for (index_t i = 0; i < nb_cells; ++i) {
    index_t counter = 0;
    const index_t nb_nodes = cells(i, cells_last);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const auto node_n = node_cellid.row(cells(i, j));
      const index_t count = node_n(node_n.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t nb = node_n(k);
        if (nb != i && i_visited(nb) != i) {
          i_visited(nb) = i;
          ++counter;
        }
      }
    }
    if (counter > max_counter)
      max_counter = counter;
  }
  return max_counter;
}
