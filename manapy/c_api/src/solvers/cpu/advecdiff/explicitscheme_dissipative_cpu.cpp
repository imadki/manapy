#include "advecdiff_compute.hpp"

#include "common/advecdiff/dissipative_common.hpp"

// CPU entry point: zero dissip_w, then accumulate every face's diffusion flux
// into it via the shared per-face routine. Serial loop, so scatter_add uses a
// plain += on the host. dissip_w is written in place.
void explicitscheme_dissipative(ArrayView<const real_t, 1> wx_face,
                                ArrayView<const real_t, 1> wy_face,
                                ArrayView<const real_t, 1> wz_face,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const real_t, 2> face_normal,
                                ArrayView<const index_t, 1> face_name,
                                ArrayView<real_t, 1> dissip_w, real_t Dxx,
                                real_t Dyy, real_t Dzz) {
  const index_t nbcell = static_cast<index_t>(dissip_w.size(0));
  for (index_t c = 0; c < nbcell; ++c)
    dissip_w(c) = real_t(0);

  const index_t nbface = static_cast<index_t>(face_cellid.size(0));
  for (index_t i = 0; i < nbface; ++i)
    dissipative_face(i, wx_face, wy_face, wz_face, face_cellid, face_normal,
                     face_name, dissip_w, Dxx, Dyy, Dzz);
}
