#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void compute_face_info_3d(ArrayView<const index_t, 2> faces,
                           ArrayView<const real_t, 2> nodes,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const real_t, 2> cell_center,
                           ArrayView<real_t, 1> face_measure,
                           ArrayView<real_t, 2> face_center,
                           ArrayView<real_t, 2> face_normal,
                           ArrayView<real_t, 2> face_tangent,
                           ArrayView<real_t, 2> face_binormal) {
  const index_t nb_faces = static_cast<index_t>(faces.size(0));
  const index_t faces_last = faces.size(1) - 1;

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t nb_vertex = faces(i, faces_last);

    const index_t i0 = faces(i, 0);
    const index_t i1 = faces(i, 1);
    const index_t i2 = faces(i, 2);

    const auto p0 = nodes.row(i0);
    const auto p1 = nodes.row(i1);
    const auto p2 = nodes.row(i2);

    real_t measure = real_t(0);
    real_t normal_storage[3] = {real_t(0), real_t(0), real_t(0)};
    ArrayView<real_t, 1> normal;
    normal.data = normal_storage;
    normal.shape[0] = 3;
    normal.stride[0] = 1;

    if (nb_vertex == 3) { // Triangle
      measure = triangle_area_3d(p0, p1, p2);
      triangle_normal_3d(p0, p1, p2, normal);
    } else if (nb_vertex == 4) { // Rectangle
      const auto p3 = nodes.row(faces(i, 3));
      measure = triangle_area_3d(p0, p1, p2) + triangle_area_3d(p0, p3, p2);
      triangle_normal_3d(p0, p1, p2, normal);
      normal(0) *= real_t(2);
      normal(1) *= real_t(2);
      normal(2) *= real_t(2);
    }

    face_measure(i) = measure;

    // Center: average of the nb_vertex points
    real_t center0 = real_t(0), center1 = real_t(0), center2 = real_t(0);
    for (index_t v = 0; v < nb_vertex; ++v) {
      const index_t nid = faces(i, v);
      center0 += nodes(nid, 0);
      center1 += nodes(nid, 1);
      center2 += nodes(nid, 2);
    }
    center0 /= real_t(nb_vertex);
    center1 /= real_t(nb_vertex);
    center2 /= real_t(nb_vertex);

    face_center(i, 0) = center0;
    face_center(i, 1) = center1;
    face_center(i, 2) = center2;

    // Face normal: orient away from the owning cell (face_cellid(i, 0)).
    const index_t cell = face_cellid(i, 0);
    const real_t snorm0 = cell_center(cell, 0) - center0;
    const real_t snorm1 = cell_center(cell, 1) - center1;
    const real_t snorm2 = cell_center(cell, 2) - center2;
    const real_t dot =
        normal(0) * snorm0 + normal(1) * snorm1 + normal(2) * snorm2;
    if (dot > real_t(0)) {
      normal(0) = -normal(0);
      normal(1) = -normal(1);
      normal(2) = -normal(2);
    }

    // 0.5 factor applies to both tetra and hexa (shared-triangle
    // decomposition above already doubled the quad's normal/area).
    face_normal(i, 0) = normal(0) * real_t(0.5);
    face_normal(i, 1) = normal(1) * real_t(0.5);
    face_normal(i, 2) = normal(2) * real_t(0.5);

    // Tangent/binormal, using the same (oriented, unscaled) normal.
    const real_t ux = nodes(i1, 0) - nodes(i0, 0);
    const real_t uy = nodes(i1, 1) - nodes(i0, 1);
    const real_t uz = nodes(i1, 2) - nodes(i0, 2);
    face_tangent(i, 0) = ux;
    face_tangent(i, 1) = uy;
    face_tangent(i, 2) = uz;

    face_binormal(i, 0) = real_t(0.5) * (uy * normal(2) - uz * normal(1));
    face_binormal(i, 1) = real_t(0.5) * (uz * normal(0) - ux * normal(2));
    face_binormal(i, 2) = real_t(0.5) * (ux * normal(1) - uy * normal(0));
  }
}
