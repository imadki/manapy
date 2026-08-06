#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void create_halo_cells(ArrayView<const index_t, 2> cells,
                        ArrayView<const index_t, 2> faces,
                        ArrayView<const index_t, 1> node_halos,
                        ArrayView<index_t, 2> node_haloid,
                        ArrayView<std::int8_t, 1> b_visited,
                        ArrayView<index_t, 2> cell_halonid,
                        ArrayView<index_t, 1> face_haloid) {
  const index_t nb_cells = static_cast<index_t>(cells.size(0));
  const index_t nb_faces = static_cast<index_t>(faces.size(0));
  const index_t node_haloid_last = node_haloid.size(1) - 1;
  const index_t cells_last = cells.size(1) - 1;
  const index_t faces_last = faces.size(1) - 1;
  const index_t cell_halonid_last = cell_halonid.size(1) - 1;

  const index_t nb_node_halos = static_cast<index_t>(node_halos.size(0));
  for (index_t i = 0; i < nb_node_halos; i += 2) {
    const index_t node_id = node_halos(i);
    const index_t halo_id = node_halos(i + 1);
    const index_t size = node_haloid(node_id, node_haloid_last);
    node_haloid(node_id, size) = halo_id;
    node_haloid(node_id, node_haloid_last) += 1;
  }

  index_t intersect_storage[1] = {-1};
  ArrayView<index_t, 1> intersect_cell;
  intersect_cell.data = intersect_storage;
  intersect_cell.shape[0] = 1;
  intersect_cell.stride[0] = 1;

  for (index_t i = 0; i < nb_faces; ++i) {
    const std::int8_t nb_nodes =
        static_cast<std::int8_t>(faces(i, faces_last));
    intersect_common(faces.row(i), nb_nodes, node_haloid.as_const(),
                      b_visited, intersect_cell);
    face_haloid(i) = intersect_cell(0);
  }

  for (index_t i = 0; i < nb_cells; ++i) {
    const index_t nb_nodes = cells(i, cells_last);
    for (index_t j = 0; j < nb_nodes; ++j) {
      const index_t node = cells(i, j);
      const auto n_halo = node_haloid.row(node).as_const();
      const index_t count = n_halo(n_halo.size(0) - 1);
      for (index_t k = 0; k < count; ++k) {
        const index_t candidate = n_halo(k);
        if (is_in_array(cell_halonid.row(i).as_const(), candidate) == -1) {
          const index_t size = cell_halonid(i, cell_halonid_last);
          cell_halonid(i, cell_halonid_last) += 1;
          cell_halonid(i, size) = candidate;
        }
      }
    }
  }
}
