#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Explicit CFL time-step (translation of _time_step). The Python original
// starts dt at 1e6 and, for every cell, lowers it to cfl * volume / lambda
// where lambda is the sum over the cell's faces of |u.n| (the convective
// speed through each face); cells with lambda == 0 are skipped. That reduction
// is a plain min over per-cell candidates, so the shared piece is the per-cell
// candidate below and the CPU/GPU drivers just take the min.

// Initial / "no limit" value: matches the Python dt = 1e6 seed. A cell with no
// convective speed returns this so it never lowers the running minimum.
inline constexpr real_t TIME_STEP_NO_LIMIT = real_t(1e6);

// std::fabs is avoided so the routine stays usable from both host and device
// code without pulling in <cmath>.
MANAPY_COMPUTE_HOST_DEVICE
real_t time_step_abs(real_t a) { return a < real_t(0) ? -a : a; }

// CFL time-step candidate for cell i: cfl * cell_volume(i) / lambda, or
// TIME_STEP_NO_LIMIT when lambda == 0. The last column of cell_faceid holds the
// number of valid faces in that row. face_normal must be at least 3 columns
// wide (the z component is read; it is 0 in 2D). Shared verbatim by the CPU
// loop and the CUDA kernel.
MANAPY_COMPUTE_HOST_DEVICE
real_t time_step_cell(index_t i, ArrayView<const real_t, 1> u,
                      ArrayView<const real_t, 1> v,
                      ArrayView<const real_t, 1> w, real_t cfl,
                      ArrayView<const real_t, 2> face_normal,
                      ArrayView<const real_t, 1> cell_volume,
                      ArrayView<const index_t, 2> cell_faceid) {
  const index_t nf = cell_faceid(i, cell_faceid.size(1) - 1);

  real_t lam = real_t(0);
  for (index_t j = 0; j < nf; ++j) {
    const index_t f = cell_faceid(i, j);
    lam += time_step_abs(u(i) * face_normal(f, 0) + v(i) * face_normal(f, 1) +
                         w(i) * face_normal(f, 2));
  }

  if (lam != real_t(0))
    return cfl * cell_volume(i) / lam;
  return TIME_STEP_NO_LIMIT;
}
