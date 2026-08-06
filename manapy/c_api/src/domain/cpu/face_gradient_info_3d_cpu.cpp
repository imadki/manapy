#include "domain_compute.hpp"

#include <stdexcept>

void face_gradient_info_3d(ArrayView<const index_t, 2> face_cellid,
                            ArrayView<const index_t, 2> faces,
                            ArrayView<const index_t, 1> face_to_phyid,
                            ArrayView<const real_t, 2> ghost_info_flt,
                            ArrayView<const index_t, 1> face_name,
                            ArrayView<const real_t, 2> face_normal,
                            ArrayView<const real_t, 2> cell_center,
                            ArrayView<const real_t, 2> halo_centvol,
                            ArrayView<const index_t, 1> face_haloid,
                            ArrayView<const real_t, 2> nodes,
                            ArrayView<real_t, 1> face_air_diamond,
                            ArrayView<real_t, 1> face_param1,
                            ArrayView<real_t, 1> face_param2,
                            ArrayView<real_t, 1> face_param3,
                            ArrayView<real_t, 2> face_f1,
                            ArrayView<real_t, 2> face_f2,
                            ArrayView<const real_t, 2> cell_shift) {
  const index_t nb_faces = static_cast<index_t>(face_cellid.size(0));
  const index_t faces_last = faces.size(1) - 1;

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t c_left = face_cellid(i, 0);
    const index_t c_right = face_cellid(i, 1);

    const index_t i_1 = faces(i, 0);
    const index_t i_2 = faces(i, 1);
    const index_t i_3 = faces(i, 2);
    index_t i_4 = i_3;
    if (faces(i, faces_last) == 4)
      i_4 = faces(i, 3);

    const real_t v1x = cell_center(c_left, 0);
    const real_t v1y = cell_center(c_left, 1);
    const real_t v1z = cell_center(c_left, 2);

    real_t v2x, v2y, v2z;
    const index_t name = face_name(i);
    if (name == 0) {
      v2x = cell_center(c_right, 0);
      v2y = cell_center(c_right, 1);
      v2z = cell_center(c_right, 2);
    } else if (name == 11 || name == 22) {
      v2x = cell_center(c_right, 0) + cell_shift(c_right, 0);
      v2y = cell_center(c_right, 1);
      v2z = cell_center(c_right, 2);
    } else if (name == 33 || name == 44) {
      v2x = cell_center(c_right, 0);
      v2y = cell_center(c_right, 1) + cell_shift(c_right, 1);
      v2z = cell_center(c_right, 2);
    } else if (name == 55 || name == 66) {
      v2x = cell_center(c_right, 0);
      v2y = cell_center(c_right, 1);
      v2z = cell_center(c_right, 2) + cell_shift(c_right, 2);
    } else if (name == 10) {
      const index_t h = face_haloid(i);
      v2x = halo_centvol(h, 0);
      v2y = halo_centvol(h, 1);
      v2z = halo_centvol(h, 2);
    } else if (face_to_phyid(i) != -1) {
      const index_t ghost_id = face_to_phyid(i);
      v2x = ghost_info_flt(ghost_id, 0);
      v2y = ghost_info_flt(ghost_id, 1);
      v2z = ghost_info_flt(ghost_id, 2);
    } else {
      throw std::runtime_error("face_gradient_info_3d: face_to_phyid[i]");
    }

    const real_t n2x = nodes(i_2, 0), n2y = nodes(i_2, 1), n2z = nodes(i_2, 2);
    const real_t n4x = nodes(i_4, 0), n4y = nodes(i_4, 1), n4z = nodes(i_4, 2);
    const real_t n1x = nodes(i_1, 0), n1y = nodes(i_1, 1), n1z = nodes(i_1, 2);
    const real_t n3x = nodes(i_3, 0), n3y = nodes(i_3, 1), n3z = nodes(i_3, 2);

    // s1 = v2 - n2; s2 = n4 - n2; s3 = v1 - n2
    const real_t s1x = v2x - n2x, s1y = v2y - n2y, s1z = v2z - n2z;
    const real_t s2x = n4x - n2x, s2y = n4y - n2y, s2z = n4z - n2z;
    const real_t s3x = v1x - n2x, s3y = v1y - n2y, s3z = v1z - n2z;

    const real_t c12x = s1y * s2z - s1z * s2y;
    const real_t c12y = s1z * s2x - s1x * s2z;
    const real_t c12z = s1x * s2y - s1y * s2x;

    const real_t c23x = s2y * s3z - s2z * s3y;
    const real_t c23y = s2z * s3x - s2x * s3z;
    const real_t c23z = s2x * s3y - s2y * s3x;

    const real_t f1x = real_t(0.5) * (c12x + c23x);
    const real_t f1y = real_t(0.5) * (c12y + c23y);
    const real_t f1z = real_t(0.5) * (c12z + c23z);

    face_f1(i, 0) = f1x;
    face_f1(i, 1) = f1y;
    face_f1(i, 2) = f1z;

    // s4 = v2 - n3; s5 = n1 - n3; s6 = v1 - n3
    const real_t s4x = v2x - n3x, s4y = v2y - n3y, s4z = v2z - n3z;
    const real_t s5x = n1x - n3x, s5y = n1y - n3y, s5z = n1z - n3z;
    const real_t s6x = v1x - n3x, s6y = v1y - n3y, s6z = v1z - n3z;

    const real_t c45x = s4y * s5z - s4z * s5y;
    const real_t c45y = s4z * s5x - s4x * s5z;
    const real_t c45z = s4x * s5y - s4y * s5x;

    const real_t c56x = s5y * s6z - s5z * s6y;
    const real_t c56y = s5z * s6x - s5x * s6z;
    const real_t c56z = s5x * s6y - s5y * s6x;

    const real_t f2x = real_t(0.5) * (c45x + c56x);
    const real_t f2y = real_t(0.5) * (c45y + c56y);
    const real_t f2z = real_t(0.5) * (c45z + c56z);

    face_f2(i, 0) = f2x;
    face_f2(i, 1) = f2y;
    face_f2(i, 2) = f2z;

    const real_t s7x = v2x - v1x, s7y = v2y - v1y, s7z = v2z - v1z;
    const real_t n0 = face_normal(i, 0);
    const real_t n1c = face_normal(i, 1);
    const real_t n2c = face_normal(i, 2);

    const real_t diamond = n0 * s7x + n1c * s7y + n2c * s7z;
    face_air_diamond(i) = diamond;

    if (diamond == real_t(0))
      throw std::runtime_error("face_gradient_info_3d: div 0");

    face_param1(i) = (f1x * n0 + f1y * n1c + f1z * n2c) / diamond;
    face_param2(i) = (f2x * n0 + f2y * n1c + f2z * n2c) / diamond;
    face_param3(i) = (n0 * n0 + n1c * n1c + n2c * n2c) / diamond;
  }
}
