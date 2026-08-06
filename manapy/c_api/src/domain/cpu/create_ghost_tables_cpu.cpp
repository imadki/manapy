#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

/*
* node_ghostid # [indices point to ghost_info_int for node neighboring ghost cells]
* cell_ghostid # [indices point to ghost_info_int for cell neighboring ghost cells]
*/
void create_ghost_tables(ArrayView<const index_t, 2> ghost_info_int,
                          ArrayView<const index_t, 2> faces,
                          ArrayView<const index_t, 2> cell_faceid,
                          ArrayView<const index_t, 2> node_cellid,
                          ArrayView<index_t, 1> ghost_i_visited,
                          ArrayView<index_t, 2> node_ghostid,
                          ArrayView<index_t, 2> cell_ghostid) {
  const index_t nb = static_cast<index_t>(ghost_info_int.size(0));
  const index_t faces_last = faces.size(1) - 1;
  const index_t node_ghostid_last = node_ghostid.size(1) - 1;
  const index_t cell_ghostid_last = cell_ghostid.size(1) - 1;

  for (index_t i = 0; i < nb; ++i) {
    const index_t bc = ghost_info_int(i, 0);
    if (bc == -1) // periodic face (see create_ghost_info): nothing to add
      continue;
    const index_t bf = ghost_info_int(i, 1);
    const index_t fid = cell_faceid(bc, bf);

    const index_t nb_face_nodes = faces(fid, faces_last);
    for (index_t j = 0; j < nb_face_nodes; ++j) {
      const index_t nid = faces(fid, j);

      const index_t size = node_ghostid(nid, node_ghostid_last);
      node_ghostid(nid, node_ghostid_last) += 1;
      node_ghostid(nid, size) = i;

      const auto neighbors = node_cellid.row(nid);
      const index_t count = neighbors(neighbors.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t cell_id = neighbors(k);
        if (ghost_i_visited(cell_id) != i) {
          ghost_i_visited(cell_id) = i;
          const index_t csize = cell_ghostid(cell_id, cell_ghostid_last);
          cell_ghostid(cell_id, cell_ghostid_last) += 1;
          cell_ghostid(cell_id, csize) = i;
        }
      }
    }
  }
}
