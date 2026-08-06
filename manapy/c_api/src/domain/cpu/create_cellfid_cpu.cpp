#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void create_cellfid(ArrayView<const index_t, 2> cells,
                     ArrayView<const index_t, 2> node_cellid,
                     ArrayView<const std::int8_t, 1> cell_type,
                     ArrayView<index_t, 2> cell_cellfid) {
  // Bounds from create_cell_faces' supported cell types: hexahedron has
  // the most faces (6), quad faces (rectangle/hex/pyramid) the most nodes
  // per face (4).
  constexpr index_t max_cell_faces = 6;
  constexpr index_t max_face_nodes = 4;

  const index_t cell_cellfid_last = cell_cellfid.size(1) - 1;
  const index_t nb_cells = static_cast<index_t>(cells.size(0));

  // Each cell only ever writes cell_cellfid(i, ...) -- its own row -- and
  // only reads const arrays, so cells are independent of one another.
  // Scratch is declared fresh inside the loop body (thread-private under
  // OpenMP with no extra annotation needed) instead of being a
  // caller-provided shared buffer, mirroring the Python original's
  // numba.prange, which allocates its own local arrays per iteration.
#pragma omp parallel for
  for (index_t i = 0; i < nb_cells; ++i) {
    index_t tmp_cell_faces_storage[max_cell_faces * max_face_nodes];
    ArrayView<index_t, 2> tmp_cell_faces;
    tmp_cell_faces.data = tmp_cell_faces_storage;
    tmp_cell_faces.shape[0] = max_cell_faces;
    tmp_cell_faces.shape[1] = max_face_nodes;
    tmp_cell_faces.stride[0] = max_face_nodes;
    tmp_cell_faces.stride[1] = 1;

    index_t tmp_size_storage[max_cell_faces + 1];
    ArrayView<index_t, 1> tmp_size_info;
    tmp_size_info.data = tmp_size_storage;
    tmp_size_info.shape[0] = max_cell_faces + 1;
    tmp_size_info.stride[0] = 1;

    index_t intersect_storage[2] = {-1, -1};
    ArrayView<index_t, 1> intersect_cells;
    intersect_cells.data = intersect_storage;
    intersect_cells.shape[0] = 2;
    intersect_cells.stride[0] = 1;

    create_cell_faces(cells.row(i), tmp_cell_faces, tmp_size_info,
                       static_cast<index_t>(cell_type(i)));

    const index_t nb_cell_faces = tmp_size_info(max_cell_faces);
    for (index_t j = 0; j < nb_cell_faces; ++j) {
      intersect_face_nodes(tmp_cell_faces.row(j).as_const(),
                            tmp_size_info(j), node_cellid, intersect_cells);

      // The face has at most two neighbors; swap so intersect_cells(0) is
      // always cells(i)'s own id.
      if (intersect_cells(1) == i) {
        intersect_cells(1) = intersect_cells(0);
        intersect_cells(0) = i;
      }

      if (intersect_cells(1) != -1) {
        const index_t size = cell_cellfid(i, cell_cellfid_last);
        cell_cellfid(i, size) = intersect_cells(1);
        cell_cellfid(i, cell_cellfid_last) += 1;
      }
    }
  }
}
