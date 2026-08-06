#pragma once

#include "array_view.hpp"
#include "common/advecdiff/convective_common.hpp"
#include "precision.hpp"

// advecdiff explicit convective residual for 3D linear advection (translation
// of _explicitscheme_convective_3d). Same as the 2D version
// (explicitscheme_convective_2d_common.hpp) but with the full three-component
// reconstruction and a third periodic axis (face names 55/66 shift z). Flux is
// the compile-time FluxScheme template; order == 1 skips the reconstruction and
// all its loads. `face_normal` must be at least 3 columns wide.
template <ConvectiveFaceKind Kind, FluxScheme Scheme>
MANAPY_COMPUTE_HOST_DEVICE void explicitscheme_convective_3d_face(
    index_t i, ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
    ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<const real_t, 1> wx_halo, ArrayView<const real_t, 1> wy_halo,
    ArrayView<const real_t, 1> wz_halo, ArrayView<const real_t, 1> psi,
    ArrayView<const real_t, 1> psi_halo, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const real_t, 2> cell_shift, index_t order) {
  const index_t cell_left = face_cellid(i, 0);

  // Left (owner) state. Reconstruction (and its loads) skipped at order 1.
  real_t w_l = w_c(cell_left);
  if (order != index_t(1)) {
    const real_t r_l0 = face_center(i, 0) - cell_center(cell_left, 0);
    const real_t r_l1 = face_center(i, 1) - cell_center(cell_left, 1);
    const real_t r_l2 = face_center(i, 2) - cell_center(cell_left, 2);
    w_l += static_cast<real_t>(order - 1) * psi(cell_left) *
           (w_x(cell_left) * r_l0 + w_y(cell_left) * r_l1 +
            w_z(cell_left) * r_l2);
  }

  // Right (far-side) state. Source varies by face kind.
  real_t w_r;
  if constexpr (Kind == ConvectiveFaceKind::Boundary) {
    w_r = w_ghost(i); // ghost value, never reconstructed
  } else if constexpr (Kind == ConvectiveFaceKind::Halo) {
    const index_t h = face_haloid(i);
    w_r = w_halo(h);
    if (order != index_t(1)) {
      const real_t r_r0 = face_center(i, 0) - halo_centvol(h, 0);
      const real_t r_r1 = face_center(i, 1) - halo_centvol(h, 1);
      const real_t r_r2 = face_center(i, 2) - halo_centvol(h, 2);
      w_r += static_cast<real_t>(order - 1) * psi_halo(h) *
             (wx_halo(h) * r_r0 + wy_halo(h) * r_r1 + wz_halo(h) * r_r2);
    }
  } else { // Inner or Periodic: neighbour cell across the face.
    const index_t cell_right = face_cellid(i, 1);
    w_r = w_c(cell_right);
    if (order != index_t(1)) {
      real_t r_r0 = face_center(i, 0) - cell_center(cell_right, 0);
      real_t r_r1 = face_center(i, 1) - cell_center(cell_right, 1);
      real_t r_r2 = face_center(i, 2) - cell_center(cell_right, 2);
      if constexpr (Kind == ConvectiveFaceKind::Periodic) {
        const index_t name = face_name(i);
        if (name == index_t(11) || name == index_t(22)) {
          r_r0 -= cell_shift(cell_right, 0);
        } else if (name == index_t(33) || name == index_t(44)) {
          r_r1 -= cell_shift(cell_right, 1);
        } else if (name == index_t(55) || name == index_t(66)) {
          r_r2 -= cell_shift(cell_right, 2);
        }
      }
      w_r += static_cast<real_t>(order - 1) * psi(cell_right) *
             (w_x(cell_right) * r_r0 + w_y(cell_right) * r_r1 +
              w_z(cell_right) * r_r2);
    }
  }

  const real_t flux = numerical_flux<Scheme>(w_l, w_r, u_face(i), v_face(i),
                                             w_face(i), face_normal.row(i));

  scatter_add(rez_w, cell_left, -flux);
  if constexpr (Kind == ConvectiveFaceKind::Inner) {
    scatter_add(rez_w, face_cellid(i, 1), flux);
  }
}
