#include "domain_compute.hpp"

#include <cmath>

void compute_face_info_2d(ArrayView<const index_t, 2> faces,
                           ArrayView<const real_t, 2> nodes,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const real_t, 2> cell_center,
                           ArrayView<real_t, 1> face_measure,
                           ArrayView<real_t, 2> face_center,
                           ArrayView<real_t, 2> face_normal) {
  const index_t nb_faces = static_cast<index_t>(faces.size(0));

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t n0 = faces(i, 0);
    const index_t n1 = faces(i, 1);

    const real_t ux = nodes(n0, 0) - nodes(n1, 0);
    const real_t uy = nodes(n0, 1) - nodes(n1, 1);
    face_measure(i) = std::sqrt(ux * ux + uy * uy);

    const real_t cx = real_t(0.5) * (nodes(n0, 0) + nodes(n1, 0));
    const real_t cy = real_t(0.5) * (nodes(n0, 1) + nodes(n1, 1));
    face_center(i, 0) = cx;
    face_center(i, 1) = cy;
    face_center(i, 2) = real_t(0);

    real_t nx = -uy;
    real_t ny = ux;
    const index_t left = face_cellid(i, 0);
    const real_t sx = cell_center(left, 0) - cx;
    const real_t sy = cell_center(left, 1) - cy;
    if (nx * sx + ny * sy > real_t(0)) {
      nx = -nx;
      ny = -ny;
    }

    face_normal(i, 0) = nx;
    face_normal(i, 1) = ny;
    face_normal(i, 2) = real_t(0);
  }
}
