#pragma once

#include "array_view.hpp"
#include "common/advec/convective_common.hpp"
#include "common/advec/numerical_flux_common.hpp"
#include "precision.hpp"

// Explicit finite-volume convective residual for 2D linear advection
// (translation of _explicitscheme_convective_2d). The Python original sweeps
// four disjoint face lists -- interior, periodic-boundary, halo and physical
// boundary faces -- that share the same MUSCL reconstruction + numerical flux
// but differ in where the right (far-side) state comes from and how the flux
// is scattered back into the per-cell residual rez_w (ConvectiveFaceKind lives
// in convective_common.hpp; scatter_add in common/helpers/scatter.hpp). Those
// four variants are expressed here with a single template parameter so the
// exact same arithmetic runs on host and GPU (MANAPY_COMPUTE_HOST_DEVICE).

// Residual contribution of a single face `i` (an entry of one of the four face
// lists). Reconstructs the left/right states to the face centre with the
// least-squares gradient (limited by psi; the (order - 1) factor drops the
// reconstruction for first-order runs), evaluates the numerical flux and
// scatters it into rez_w: subtracted from the owner cell and, for interior
// faces only, added to the neighbour. `face_normal` must be at least 3 columns
// wide (numerical_flux reads the z component). w_z/wz_halo are unused in 2D and
// are not passed here.
template <ConvectiveFaceKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void explicitscheme_convective_2d_face(
    index_t i, ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
    ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> wx_halo,
    ArrayView<const real_t, 1> wy_halo, ArrayView<const real_t, 1> psi,
    ArrayView<const real_t, 1> psi_halo, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const real_t, 2> cell_shift,
    index_t order,
    index_t scheme) {
  const real_t recon = static_cast<real_t>(order - 1);

  // Left (owner) state, reconstructed to the face centre.
  const index_t cell_left = face_cellid(i, 0);
  const real_t r_l0 = face_center(i, 0) - cell_center(cell_left, 0);
  const real_t r_l1 = face_center(i, 1) - cell_center(cell_left, 1);
  const real_t w_l =
      w_c(cell_left) +
      recon * psi(cell_left) * (w_x(cell_left) * r_l0 + w_y(cell_left) * r_l1);

  // Right (far-side) state. Where it comes from is the only thing that varies
  // between the face kinds.
  real_t w_r;
  if constexpr (Kind == ConvectiveFaceKind::Boundary) {
    // Ghost value, no reconstruction (matches `w_r = w_r` in the original).
    w_r = w_ghost(i);
  } else if constexpr (Kind == ConvectiveFaceKind::Halo) {
    const index_t h = face_haloid(i);
    const real_t r_r0 = face_center(i, 0) - halo_centvol(h, 0);
    const real_t r_r1 = face_center(i, 1) - halo_centvol(h, 1);
    w_r = w_halo(h) +
          recon * psi_halo(h) * (wx_halo(h) * r_r0 + wy_halo(h) * r_r1);
  } else { // Inner or Periodic: neighbour cell across the face.
    const index_t cell_right = face_cellid(i, 1);
    real_t r_r0 = face_center(i, 0) - cell_center(cell_right, 0);
    real_t r_r1 = face_center(i, 1) - cell_center(cell_right, 1);
    if constexpr (Kind == ConvectiveFaceKind::Periodic) {
      // The neighbour lives across a periodic boundary; shift it back into the
      // owner's frame along the wrapped axis (11/22 = x, 33/44 = y).
      const index_t name = face_name(i);
      if (name == index_t(11) || name == index_t(22)) {
        r_r0 -= cell_shift(cell_right, 0);
      } else if (name == index_t(33) || name == index_t(44)) {
        r_r1 -= cell_shift(cell_right, 1);
      }
    }
    w_r = w_c(cell_right) + recon * psi(cell_right) *
                                (w_x(cell_right) * r_r0 + w_y(cell_right) * r_r1);
  }

  const real_t flux = numerical_flux(scheme, w_l, w_r, u_face(i), v_face(i),
                                     w_face(i), face_normal.row(i));

  scatter_add(rez_w, cell_left, -flux);
  if constexpr (Kind == ConvectiveFaceKind::Inner) {
    scatter_add(rez_w, face_cellid(i, 1), flux);
  }
}
