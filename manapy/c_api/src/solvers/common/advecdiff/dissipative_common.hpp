#pragma once

#include "array_view.hpp"
#include "common/helpers/scatter.hpp"
#include "precision.hpp"

// advecdiff diffusion (dissipative) flux for a single face (translation of
// _explicitscheme_dissipative in to_convert.py). Shared verbatim by the CPU
// loop and the CUDA kernel (MANAPY_COMPUTE_HOST_DEVICE). Dimension-agnostic: it
// reads all three normal/face-gradient components, which are 0 in the unused
// axis for 2D, so one routine serves both 2D and 3D.
//
// q = Dxx*wx_face*n0 + Dyy*wy_face*n1 + Dzz*wz_face*n2 is scattered into
// dissip_w: the owner cell gets +q, and for an interior face (face_name == 0)
// the neighbour gets -q. (Note the sign is the opposite of the convective
// residual's scatter.) `face_normal` must be at least 3 columns wide.
MANAPY_COMPUTE_HOST_DEVICE
void dissipative_face(index_t i, ArrayView<const real_t, 1> wx_face,
                      ArrayView<const real_t, 1> wy_face,
                      ArrayView<const real_t, 1> wz_face,
                      ArrayView<const index_t, 2> face_cellid,
                      ArrayView<const real_t, 2> face_normal,
                      ArrayView<const index_t, 1> face_name,
                      ArrayView<real_t, 1> dissip_w, real_t Dxx, real_t Dyy,
                      real_t Dzz) {
  const real_t q = Dxx * wx_face(i) * face_normal(i, 0) +
                   Dyy * wy_face(i) * face_normal(i, 1) +
                   Dzz * wz_face(i) * face_normal(i, 2);

  scatter_add(dissip_w, face_cellid(i, 0), q);
  if (face_name(i) == index_t(0)) { // interior face: neighbour gets -q
    scatter_add(dissip_w, face_cellid(i, 1), -q);
  }
}
