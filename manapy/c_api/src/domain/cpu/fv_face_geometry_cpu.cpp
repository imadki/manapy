#include "domain_compute.hpp"

#include <cmath>
#include <stdexcept>

void fv_face_geometry(ArrayView<const index_t, 2> face_cellid,
                       ArrayView<const index_t, 1> face_name,
                       ArrayView<const real_t, 2> face_normal,
                       ArrayView<const real_t, 2> face_center,
                       ArrayView<const index_t, 1> face_haloid,
                       ArrayView<const real_t, 2> cell_center,
                       ArrayView<const real_t, 2> halo_centvol,
                       ArrayView<const real_t, 2> cell_shift,
                       ArrayView<real_t, 1> fv_coeff,
                       ArrayView<real_t, 1> fv_corrx,
                       ArrayView<real_t, 1> fv_corry,
                       ArrayView<real_t, 1> fv_corrz,
                       ArrayView<real_t, 1> fv_weight_left) {
  const index_t nb_faces = static_cast<index_t>(face_cellid.size(0));

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t c_left = face_cellid(i, 0);
    const index_t c_right = face_cellid(i, 1);

    const real_t left_x = cell_center(c_left, 0);
    const real_t left_y = cell_center(c_left, 1);
    const real_t left_z = cell_center(c_left, 2);

    real_t right_x = face_center(i, 0);
    real_t right_y = face_center(i, 1);
    real_t right_z = face_center(i, 2);
    bool has_right = false;

    const real_t nx = face_normal(i, 0);
    const real_t ny = face_normal(i, 1);
    const real_t nz = face_normal(i, 2);

    const index_t name = face_name(i);
    if (name == 0) {
      right_x = cell_center(c_right, 0);
      right_y = cell_center(c_right, 1);
      right_z = cell_center(c_right, 2);
      has_right = true;
    } else if (name == 11 || name == 22) {
      right_x = cell_center(c_right, 0) + cell_shift(c_right, 0);
      right_y = cell_center(c_right, 1);
      right_z = cell_center(c_right, 2);
      has_right = true;
    } else if (name == 33 || name == 44) {
      right_x = cell_center(c_right, 0);
      right_y = cell_center(c_right, 1) + cell_shift(c_right, 1);
      right_z = cell_center(c_right, 2);
      has_right = true;
    } else if (name == 55 || name == 66) {
      right_x = cell_center(c_right, 0);
      right_y = cell_center(c_right, 1);
      right_z = cell_center(c_right, 2) + cell_shift(c_right, 2);
      has_right = true;
    } else if (name == 10) {
      const index_t h = face_haloid(i);
      right_x = halo_centvol(h, 0);
      right_y = halo_centvol(h, 1);
      right_z = halo_centvol(h, 2);
      has_right = true;
    }

    const real_t dx = right_x - left_x;
    const real_t dy = right_y - left_y;
    const real_t dz = right_z - left_z;
    const real_t sfd = nx * dx + ny * dy + nz * dz;
    if (sfd == real_t(0))
      throw std::runtime_error(
          "fv_face_geometry: zero projected face distance in FV-like "
          "geometry");

    const real_t nsq = nx * nx + ny * ny + nz * nz;
    real_t abs_sfd = sfd;
    if (abs_sfd < real_t(0))
      abs_sfd = -abs_sfd;
    fv_coeff(i) = nsq / abs_sfd;

    const real_t signed_coeff = nsq / sfd;
    fv_corrx(i) = nx - signed_coeff * dx;
    fv_corry(i) = ny - signed_coeff * dy;
    fv_corrz(i) = nz - signed_coeff * dz;

    if (has_right) {
      const real_t dlx = face_center(i, 0) - left_x;
      const real_t dly = face_center(i, 1) - left_y;
      const real_t dlz = face_center(i, 2) - left_z;
      const real_t drx = right_x - face_center(i, 0);
      const real_t dry = right_y - face_center(i, 1);
      const real_t drz = right_z - face_center(i, 2);
      const real_t dleft = std::sqrt(dlx * dlx + dly * dly + dlz * dlz);
      const real_t dright = std::sqrt(drx * drx + dry * dry + drz * drz);
      const real_t dist = dleft + dright;
      if (dist == real_t(0))
        fv_weight_left(i) = real_t(0.5);
      else
        fv_weight_left(i) = dright / dist;
    } else {
      fv_weight_left(i) = real_t(1);
    }
  }
}
