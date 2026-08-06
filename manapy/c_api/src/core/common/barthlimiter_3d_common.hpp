#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// std::min/std::max/std::fabs are avoided here so the routine stays usable
// from both host and device code without pulling in <algorithm>/<cmath>.
// Named distinctly from common/barthlimiter_2d_common.hpp's helpers so both
// headers stay safe to include in the same translation unit.
MANAPY_COMPUTE_HOST_DEVICE
real_t barth3d_max(real_t a, real_t b) { return a > b ? a : b; }

MANAPY_COMPUTE_HOST_DEVICE
real_t barth3d_min(real_t a, real_t b) { return a < b ? a : b; }

MANAPY_COMPUTE_HOST_DEVICE
real_t barth3d_abs(real_t a) { return a < real_t(0) ? -a : a; }

// Barth-Jespersen slope limiter for a single cell i on a 3D unstructured mesh:
// shrinks the reconstructed gradient (w_x, w_y, w_z) so the linear
// extrapolation to each face stays within [w_min, w_max] of the cell's face
// neighbours (translation of _barthlimiter_3d in to_convert.py). Shared
// verbatim by the CPU loop (cpu/barthlimiter_3d_cpu.cpp) and the CUDA kernel
// (gpu/barthlimiter_3d_cuda.cu). Writes psi at index i in place.
//
// face_name selects how the far-side value is fetched: 0 or >10 is an
// interior face (both sides from w_c via face_cellid), 10 is a halo face
// (w_halo via face_haloid), anything else is a boundary face (w_ghost indexed
// by face id) -- the halo/ghost precedence is flipped relative to
// barthlimiter_2d_cell (common/barthlimiter_2d_common.hpp).
MANAPY_COMPUTE_HOST_DEVICE
void barthlimiter_3d_cell(
    index_t i, ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> cell_faceid,
    ArrayView<const index_t, 1> face_name, ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 2> cell_center, ArrayView<const real_t, 2> face_center,
    ArrayView<real_t, 1> psi) {
  constexpr real_t val = real_t(1);

  const real_t wc_i = w_c(i);
  const real_t cx = cell_center(i, 0);
  const real_t cy = cell_center(i, 1);
  const real_t cz = cell_center(i, 2);

  real_t w_max = wc_i;
  real_t w_min = wc_i;

  const index_t nf = cell_faceid(i, cell_faceid.size(1) - 1);
  for (index_t j = 0; j < nf; ++j) {
    const index_t face = cell_faceid(i, j);
    const index_t name = face_name(face);
    const real_t left = w_c(face_cellid(face, 0));

    real_t right;
    if (name == index_t(0) || name > index_t(10)) {
      right = w_c(face_cellid(face, 1));
    } else if (name == index_t(10)) {
      right = w_halo(face_haloid(face));
    } else {
      right = w_ghost(face);
    }

    w_max = barth3d_max(w_max, barth3d_max(left, right));
    w_min = barth3d_min(w_min, barth3d_min(left, right));
  }

  real_t psi_i = val;
  for (index_t j = 0; j < nf; ++j) {
    const index_t face = cell_faceid(i, j);

    const real_t r_x = face_center(face, 0) - cx;
    const real_t r_y = face_center(face, 1) - cy;
    const real_t r_z = face_center(face, 2) - cz;
    const real_t delta2 = w_x(i) * r_x + w_y(i) * r_y + w_z(i) * r_z;

    real_t psi_ij;
    if (barth3d_abs(delta2) < real_t(1e-10)) {
      psi_ij = val;
    } else if (delta2 > real_t(0)) {
      psi_ij = barth3d_min(val, (w_max - wc_i) / delta2);
    } else {
      psi_ij = barth3d_min(val, (w_min - wc_i) / delta2);
    }

    psi_i = barth3d_min(psi_i, psi_ij);
  }

  psi(i) = psi_i;
}
