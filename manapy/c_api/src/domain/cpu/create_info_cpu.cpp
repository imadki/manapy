#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void create_info(ArrayView<const index_t, 2> cells,
                  ArrayView<const index_t, 2> node_cellid,
                  ArrayView<const std::int8_t, 1> cell_type,
                  ArrayView<index_t, 2> tmp_cell_faces,
                  ArrayView<index_t, 1> tmp_size_info,
                  ArrayView<index_t, 2> tmp_cell_faces_map,
                  ArrayView<index_t, 2> faces,
                  ArrayView<index_t, 2> cell_faceid,
                  ArrayView<index_t, 2> face_cellid,
                  ArrayView<index_t, 2> cell_cellfid,
                  ArrayView<index_t, 1> faces_counter) {
  index_t intersect_storage[2] = {-1, -1};
  ArrayView<index_t, 1> intersect_cells;
  intersect_cells.data = intersect_storage;
  intersect_cells.shape[0] = 2;
  intersect_cells.stride[0] = 1;

  const index_t nb_faces = (tmp_cell_faces_map.size(1) - 1) / 2;
  const index_t map_last = tmp_cell_faces_map.size(1) - 1;
  const index_t tmp_size_last = tmp_size_info.size(0) - 1;
  const index_t faces_last = faces.size(1) - 1;
  const index_t cell_faceid_last = cell_faceid.size(1) - 1;
  const index_t cell_cellfid_last = cell_cellfid.size(1) - 1;
  const index_t nb_cells = static_cast<index_t>(cells.size(0));

  for (index_t i = 0; i < nb_cells; ++i) {
    create_cell_faces(cells.row(i), tmp_cell_faces, tmp_size_info,
                       static_cast<index_t>(cell_type(i)));

    // For every face of cells(i): get the intersection of the neighboring
    // cells of this face's nodes (N*n*n). The result is at most two cells,
    // `intersect_cells`.
    const index_t nb_cell_faces = tmp_size_info(tmp_size_last);
    for (index_t j = 0; j < nb_cell_faces; ++j) {
      intersect_face_nodes(tmp_cell_faces.row(j).as_const(),
                            tmp_size_info(j), node_cellid, intersect_cells);

      // The face has at most two neighbors; swap so intersect_cells(0) is
      // always cells(i)'s own id.
      if (intersect_cells(1) == i) {
        intersect_cells(1) = intersect_cells(0);
        intersect_cells(0) = i;
      }

      index_t face_id = -1;
      // Check whether the neighbor cell already contributed this face.
      if (intersect_cells(1) != -1) {
        const index_t map_count = tmp_cell_faces_map(i, map_last);
        for (index_t k = 0; k < map_count; ++k) {
          if (tmp_cell_faces_map(i, k) == intersect_cells(1))
            face_id = tmp_cell_faces_map(i, nb_faces + k);
        }
      }

      if (face_id == -1) {
        face_id = faces_counter(0);
        faces_counter(0) += 1;

        const index_t nb_face_nodes = tmp_size_info(j);
        for (index_t k = 0; k < nb_face_nodes; ++k)
          faces(face_id, k) = tmp_cell_faces(j, k);
        faces(face_id, faces_last) = nb_face_nodes;

        // Record this face against the neighbor cell so it can find it
        // instead of creating a duplicate when it's processed in turn.
        if (intersect_cells(1) != -1) {
          const index_t neighbor = intersect_cells(1);
          const index_t size = tmp_cell_faces_map(neighbor, map_last);
          tmp_cell_faces_map(neighbor, size) = i;
          tmp_cell_faces_map(neighbor, nb_faces + size) = face_id;
          tmp_cell_faces_map(neighbor, map_last) += 1;
        }
      }

      // (cell_faces) Create cell faces
      cell_faceid(i, j) = face_id;
      cell_faceid(i, cell_faceid_last) += 1;

      // (face_cellid) Create neighboring cells of each face
      face_cellid(face_id, 0) = intersect_cells(0);
      face_cellid(face_id, 1) = intersect_cells(1);

      // (cell_cellfid) Create neighboring cells of the cell by face
      if (intersect_cells(1) != -1) {
        const index_t size = cell_cellfid(i, cell_cellfid_last);
        cell_cellfid(i, size) = intersect_cells(1);
        cell_cellfid(i, cell_cellfid_last) += 1;
      }
    }
  }
}
