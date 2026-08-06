#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

// Any cell that has a node that has at least a neighbor phyid
void get_cell_nb_phyid(ArrayView<const index_t, 2> phy_faces,
                        ArrayView<const index_t, 2> node_cellid,
                        ArrayView<index_t, 1> i_visited,
                        ArrayView<index_t, 1> cell_nb_phyid) {
  const index_t nb_phy_faces = static_cast<index_t>(phy_faces.size(0));
  const index_t phy_last = phy_faces.size(1) - 1;

  for (index_t i = 0; i < nb_phy_faces; ++i) {
    const index_t nb_nodes = phy_faces(i, phy_last);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const index_t nid = phy_faces(i, j);
      const auto neighbors = node_cellid.row(nid);
      const index_t count = neighbors(neighbors.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t cell_id = neighbors(k);
        if (i_visited(cell_id) != i) {
          i_visited(cell_id) = i;
          cell_nb_phyid(cell_id) += 1;
        }
      }
    }
  }
}
