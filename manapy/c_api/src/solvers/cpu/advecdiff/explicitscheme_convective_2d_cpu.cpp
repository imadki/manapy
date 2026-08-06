#include "advecdiff_compute.hpp"

#include "common/advecdiff/explicitscheme_convective_2d_common.hpp"

namespace {

// One flux scheme, resolved at compile time. Zero the residual, then sweep the
// four face lists in the Python order, accumulating into rez_w. Serial loop, so
// the scatter needs no atomics (scatter_add uses a plain += on the host).
template <FluxScheme Scheme>
void run(ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
         ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
         ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
         ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
         ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> wx_halo,
         ArrayView<const real_t, 1> wy_halo, ArrayView<const real_t, 1> psi,
         ArrayView<const real_t, 1> psi_halo,
         ArrayView<const real_t, 2> cell_center,
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
         ArrayView<const real_t, 2> cell_shift, index_t order) {
  const index_t nbcell = static_cast<index_t>(rez_w.size(0));
  for (index_t c = 0; c < nbcell; ++c)
    rez_w(c) = real_t(0);

  const index_t n_inner = static_cast<index_t>(d_innerfaces.size(0));
  for (index_t k = 0; k < n_inner; ++k)
    explicitscheme_convective_2d_face<ConvectiveFaceKind::Inner, Scheme>(
        d_innerfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, wx_halo, wy_halo, psi, psi_halo, cell_center, face_center,
        halo_centvol, face_cellid, face_normal, face_haloid, face_name,
        cell_shift, order);

  const index_t n_periodic =
      static_cast<index_t>(d_periodicboundaryfaces.size(0));
  for (index_t k = 0; k < n_periodic; ++k)
    explicitscheme_convective_2d_face<ConvectiveFaceKind::Periodic, Scheme>(
        d_periodicboundaryfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face,
        w_face, w_x, w_y, wx_halo, wy_halo, psi, psi_halo, cell_center,
        face_center, halo_centvol, face_cellid, face_normal, face_haloid,
        face_name, cell_shift, order);

  const index_t n_halo = static_cast<index_t>(d_halofaces.size(0));
  for (index_t k = 0; k < n_halo; ++k)
    explicitscheme_convective_2d_face<ConvectiveFaceKind::Halo, Scheme>(
        d_halofaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, wx_halo, wy_halo, psi, psi_halo, cell_center, face_center,
        halo_centvol, face_cellid, face_normal, face_haloid, face_name,
        cell_shift, order);

  const index_t n_boundary = static_cast<index_t>(d_boundaryfaces.size(0));
  for (index_t k = 0; k < n_boundary; ++k)
    explicitscheme_convective_2d_face<ConvectiveFaceKind::Boundary, Scheme>(
        d_boundaryfaces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face,
        w_x, w_y, wx_halo, wy_halo, psi, psi_halo, cell_center, face_center,
        halo_centvol, face_cellid, face_normal, face_haloid, face_name,
        cell_shift, order);
}

} // namespace

// CPU entry point. `scheme` selects the flux once here (a single switch),
// dispatching to the compile-time-specialised run<FluxScheme>; the hot loops
// carry no per-face flux branch. rez_w is written in place.
void explicitscheme_convective_2d(
    ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
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
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_periodicboundaryfaces,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme) {
#define MANAPY_ADVECDIFF_RUN(S)                                                 \
  run<S>(rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face, w_x, w_y,         \
         wx_halo, wy_halo, psi, psi_halo, cell_center, face_center,            \
         halo_centvol, face_cellid, face_normal, face_haloid, face_name,       \
         d_innerfaces, d_halofaces, d_boundaryfaces, d_periodicboundaryfaces,  \
         cell_shift, order)

  switch (scheme) {
  case index_t(0):
    MANAPY_ADVECDIFF_RUN(FluxScheme::Upwind);
    break;
  case index_t(1):
    MANAPY_ADVECDIFF_RUN(FluxScheme::Centered);
    break;
  case index_t(3):
    MANAPY_ADVECDIFF_RUN(FluxScheme::LaxFriedrichs);
    break;
  default: // 2 = rusanov, and the safe fallback
    MANAPY_ADVECDIFF_RUN(FluxScheme::Rusanov);
    break;
  }
#undef MANAPY_ADVECDIFF_RUN
}
