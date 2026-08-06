#include "domain_compute.hpp"

void face_gradient_info_2d(ArrayView<const index_t, 2> face_cellid,
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
                            ArrayView<real_t, 1> face_param4,
                            ArrayView<real_t, 2> face_f1,
                            ArrayView<real_t, 2> face_f2,
                            ArrayView<real_t, 2> face_f3,
                            ArrayView<real_t, 2> face_f4,
                            ArrayView<const real_t, 2> cell_shift) {
  const index_t nb_faces = static_cast<index_t>(face_cellid.size(0));

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t c_left = face_cellid(i, 0);
    const index_t c_right = face_cellid(i, 1);

    const index_t i_1 = faces(i, 0);
    const index_t i_2 = faces(i, 1);

    const real_t xy_1_0 = nodes(i_1, 0);
    const real_t xy_1_1 = nodes(i_1, 1);
    const real_t xy_2_0 = nodes(i_2, 0);
    const real_t xy_2_1 = nodes(i_2, 1);

    const real_t v_1_0 = cell_center(c_left, 0);
    const real_t v_1_1 = cell_center(c_left, 1);
    real_t v_2_0 = real_t(0);
    real_t v_2_1 = real_t(0);

    const index_t name = face_name(i);
    if (name == 0) {
      v_2_0 = cell_center(c_right, 0);
      v_2_1 = cell_center(c_right, 1);
    } else if (name == 11 || name == 22) {
      v_2_0 = cell_center(c_right, 0) + cell_shift(c_right, 0);
      v_2_1 = cell_center(c_right, 1);
    } else if (name == 33 || name == 44) {
      v_2_0 = cell_center(c_right, 0);
      v_2_1 = cell_center(c_right, 1) + cell_shift(c_right, 1);
    } else if (name == 10) {
      const index_t h = face_haloid(i);
      v_2_0 = halo_centvol(h, 0);
      v_2_1 = halo_centvol(h, 1);
    } else if (face_to_phyid(i) != -1) {
      const index_t ghost_id = face_to_phyid(i);
      v_2_0 = ghost_info_flt(ghost_id, 0);
      v_2_1 = ghost_info_flt(ghost_id, 1);
    }

    const real_t f1_0 = v_1_0 - xy_1_0;
    const real_t f1_1 = v_1_1 - xy_1_1;
    const real_t f2_0 = xy_2_0 - v_1_0;
    const real_t f2_1 = xy_2_1 - v_1_1;
    const real_t f3_0 = v_2_0 - xy_2_0;
    const real_t f3_1 = v_2_1 - xy_2_1;
    const real_t f4_0 = xy_1_0 - v_2_0;
    const real_t f4_1 = xy_1_1 - v_2_1;

    face_f1(i, 0) = f1_0;
    face_f1(i, 1) = f1_1;
    face_f2(i, 0) = f2_0;
    face_f2(i, 1) = f2_1;
    face_f3(i, 0) = f3_0;
    face_f3(i, 1) = f3_1;
    face_f4(i, 0) = f4_0;
    face_f4(i, 1) = f4_1;

    const real_t n1 = face_normal(i, 0);
    const real_t n2 = face_normal(i, 1);

    const real_t air = real_t(0.5) * ((xy_2_0 - xy_1_0) * (v_2_1 - v_1_1) +
                                       (v_1_0 - v_2_0) * (xy_2_1 - xy_1_1));
    face_air_diamond(i) = air;
    const real_t inv_air2 = real_t(1) / (real_t(2) * air);

    face_param1(i) = inv_air2 * ((f1_1 + f2_1) * n1 - (f1_0 + f2_0) * n2);
    face_param2(i) = inv_air2 * ((f2_1 + f3_1) * n1 - (f2_0 + f3_0) * n2);
    face_param3(i) = inv_air2 * ((f3_1 + f4_1) * n1 - (f3_0 + f4_0) * n2);
    face_param4(i) = inv_air2 * ((f4_1 + f1_1) * n1 - (f4_0 + f1_0) * n2);
  }
}
