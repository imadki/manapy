#include "advec_compute.hpp"

#include "common/advec/explicitscheme_convective_3d_common.hpp"

// CPU entry point: zero the residual, then sweep the four face lists in the
// same order as the Python original, accumulating each face's flux into rez_w
// via the shared per-face routine. The loop is serial, so the scatter into
// rez_w needs no atomics (scatter_add uses a plain += on the host).
void explicitscheme_convective_3d(
    ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
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
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_periodicboundaryfaces,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme) {

  const index_t nbcell = static_cast<index_t>(rez_w.size(0));
  for (index_t c = 0; c < nbcell; ++c)
    rez_w(c) = real_t(0);

  const index_t n_inner = static_cast<index_t>(d_innerfaces.size(0));
  for (index_t k = 0; k < n_inner; ++k)
    explicitscheme_convective_3d_face<ConvectiveFaceKind::Inner>(
        d_innerfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, w_z, wx_halo, wy_halo, wz_halo, psi, psi_halo, cell_center,
        face_center, halo_centvol, face_cellid, face_normal, face_haloid,
        face_name, cell_shift, order, scheme);

  const index_t n_periodic =
      static_cast<index_t>(d_periodicboundaryfaces.size(0));
  for (index_t k = 0; k < n_periodic; ++k)
    explicitscheme_convective_3d_face<ConvectiveFaceKind::Periodic>(
        d_periodicboundaryfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face,
        w_face, w_x, w_y, w_z, wx_halo, wy_halo, wz_halo, psi, psi_halo,
        cell_center, face_center, halo_centvol, face_cellid, face_normal,
        face_haloid, face_name, cell_shift, order, scheme);

  const index_t n_halo = static_cast<index_t>(d_halofaces.size(0));
  for (index_t k = 0; k < n_halo; ++k)
    explicitscheme_convective_3d_face<ConvectiveFaceKind::Halo>(
        d_halofaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, w_z, wx_halo, wy_halo, wz_halo, psi, psi_halo, cell_center,
        face_center, halo_centvol, face_cellid, face_normal, face_haloid,
        face_name, cell_shift, order, scheme);

  const index_t n_boundary = static_cast<index_t>(d_boundaryfaces.size(0));
  for (index_t k = 0; k < n_boundary; ++k)
    explicitscheme_convective_3d_face<ConvectiveFaceKind::Boundary>(
        d_boundaryfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, w_z, wx_halo, wy_halo, wz_halo, psi, psi_halo, cell_center,
        face_center, halo_centvol, face_cellid, face_normal, face_haloid,
        face_name, cell_shift, order, scheme);
}
