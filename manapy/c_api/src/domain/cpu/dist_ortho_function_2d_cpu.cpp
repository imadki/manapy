#include "domain_compute.hpp"

#include <cmath>

void dist_ortho_function_2d(ArrayView<const index_t, 1> d_innerfaces,
                             ArrayView<const index_t, 1> d_boundaryfaces,
                             ArrayView<const index_t, 2> face_cellid,
                             ArrayView<const real_t, 2> cell_center,
                             ArrayView<const real_t, 2> face_center,
                             ArrayView<const real_t, 2> face_normal,
                             ArrayView<real_t, 1> face_dist_ortho) {
  const index_t nb_boundary = static_cast<index_t>(d_boundaryfaces.size(0));
  for (index_t i = 0; i < nb_boundary; ++i) {
    const index_t bf = d_boundaryfaces(i);
    const index_t k = face_cellid(bf, 0);
    const real_t u0 = face_normal(bf, 0);
    const real_t u1 = face_normal(bf, 1);

    const real_t v0 = cell_center(k, 0) - face_center(bf, 0);
    const real_t v1 = cell_center(k, 1) - face_center(bf, 1);
    const real_t dot = v0 * u0 + v1 * u1;
    const real_t projection0 = cell_center(k, 0) - dot * u0;
    const real_t projection1 = cell_center(k, 1) - dot * u1;
    const real_t dx = cell_center(k, 0) - projection0;
    const real_t dy = cell_center(k, 1) - projection1;
    face_dist_ortho(bf) = real_t(2) * std::sqrt(dx * dx + dy * dy);
  }

  const index_t nb_inner = static_cast<index_t>(d_innerfaces.size(0));
  for (index_t i = 0; i < nb_inner; ++i) {
    const index_t bf = d_innerfaces(i);
    const index_t k = face_cellid(bf, 0);
    const index_t l = face_cellid(bf, 1);
    const real_t u0 = face_normal(bf, 0);
    const real_t u1 = face_normal(bf, 1);

    const real_t v0 = cell_center(k, 0) - face_center(bf, 0);
    const real_t v1 = cell_center(k, 1) - face_center(bf, 1);
    const real_t dot = v0 * u0 + v1 * u1;
    const real_t projection0 = cell_center(k, 0) - dot * u0;
    const real_t projection1 = cell_center(k, 1) - dot * u1;
    const real_t dx = cell_center(k, 0) - projection0;
    const real_t dy = cell_center(k, 1) - projection1;

    const real_t v0_bis = cell_center(l, 0) - face_center(bf, 0);
    const real_t v1_bis = cell_center(l, 1) - face_center(bf, 1);
    const real_t dot_bis = v0_bis * u0 + v1_bis * u1;
    const real_t projection0_bis = cell_center(l, 0) - dot_bis * u0;
    const real_t projection1_bis = cell_center(l, 1) - dot_bis * u1;
    const real_t dx_bis = cell_center(l, 0) - projection0_bis;
    const real_t dy_bis = cell_center(l, 1) - projection1_bis;

    face_dist_ortho(bf) = std::sqrt(dx * dx + dy * dy) +
                           std::sqrt(dx_bis * dx_bis + dy_bis * dy_bis);
  }
}
