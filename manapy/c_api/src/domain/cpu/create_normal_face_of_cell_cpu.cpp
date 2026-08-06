#include "domain_compute.hpp"

void create_normal_face_of_cell(ArrayView<const real_t, 2> cell_center,
                                 ArrayView<const real_t, 2> face_center,
                                 ArrayView<const index_t, 2> cell_faceid,
                                 ArrayView<const real_t, 2> face_normal,
                                 ArrayView<real_t, 3> cell_nf) {
  const index_t nb_cells = static_cast<index_t>(cell_faceid.size(0));
  const index_t cell_faceid_last = cell_faceid.size(1) - 1;

  for (index_t i = 0; i < nb_cells; ++i) {
    const real_t cx = cell_center(i, 0);
    const real_t cy = cell_center(i, 1);
    const real_t cz = cell_center(i, 2);

    const index_t nb_faces = cell_faceid(i, cell_faceid_last);
    for (index_t j = 0; j < nb_faces; ++j) {
      const index_t fid = cell_faceid(i, j);
      real_t nx = face_normal(fid, 0);
      real_t ny = face_normal(fid, 1);
      real_t nz = face_normal(fid, 2);

      const real_t sx = cx - face_center(fid, 0);
      const real_t sy = cy - face_center(fid, 1);
      const real_t sz = cz - face_center(fid, 2);
      if (sx * nx + sy * ny + sz * nz > real_t(0)) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }

      cell_nf(i, j, 0) = nx;
      cell_nf(i, j, 1) = ny;
      cell_nf(i, j, 2) = nz;
    }
  }
}
