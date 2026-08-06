#pragma once

#include <cmath>

#include "array_view.hpp"
#include "precision.hpp"

// Explicit CFL time-step for pure diffusion (translation of _time_step). dt
// starts at 1e6 and, for every cell, is lowered to cfl * volume / lambda, where
// lambda is summed over the cell's faces. Each face contributes only the
// diffusion term (Dxx+Dyy+Dzz) * mes^2 / volume, with mes = ||face_normal||;
// unlike advecdiff's time step there is no convective |u.n| term (the Python
// original takes u/v/w but never reads them). Cells with lambda == 0 are
// skipped. The reduction is a plain min over per-cell candidates, so the shared
// piece is the per-cell candidate below and the CPU/GPU drivers just take the
// min.
//
// u, v, w, face_measure and dim from the Python signature are unused by the
// computation (kept only for signature parity in the binding).

// Initial / "no limit" value: matches the Python dt = 1e6 seed. A cell with no
// diffusive contribution returns this so it never lowers the running minimum.
inline constexpr real_t TIME_STEP_NO_LIMIT = real_t(1e6);

// Portable square root, usable from host and device. On the device pass nvcc
// picks the CUDA intrinsic; on the host pass it is std::sqrt (<cmath>). Kept
// faithful to the Python `mes = sqrt(n.n)` even though `mes^2` cancels the
// root, so the result is bit-identical to the NumPy original.
MANAPY_COMPUTE_HOST_DEVICE
real_t time_step_sqrt(real_t a) {
#if defined(__CUDA_ARCH__)
  return sqrt(a);
#else
  return std::sqrt(a);
#endif
}

// CFL time-step candidate for cell i: cfl * cell_volume(i) / lambda, or
// TIME_STEP_NO_LIMIT when lambda == 0. The last column of cell_faceid holds the
// number of valid faces in that row. face_normal must be at least 3 columns
// wide (the z component is read; it is 0 in 2D). Dxx/Dyy/Dzz are the
// anisotropic diffusion coefficients. Shared verbatim by the CPU loop and the
// CUDA kernel.
MANAPY_COMPUTE_HOST_DEVICE
real_t time_step_cell(index_t i, real_t cfl,
                      ArrayView<const real_t, 2> face_normal,
                      ArrayView<const real_t, 1> cell_volume,
                      ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                      real_t Dyy, real_t Dzz) {
  const index_t nf = cell_faceid(i, cell_faceid.size(1) - 1);

  real_t lam = real_t(0);
  for (index_t j = 0; j < nf; ++j) {
    const index_t f = cell_faceid(i, j);
    const real_t n0 = face_normal(f, 0);
    const real_t n1 = face_normal(f, 1);
    const real_t n2 = face_normal(f, 2);

    // Written as the Python original (three separate D*mes^2 terms, not
    // (Dxx+Dyy+Dzz)*mes^2) so the summation order -- and therefore the
    // rounding -- matches NumPy exactly.
    const real_t mes = time_step_sqrt(n0 * n0 + n1 * n1 + n2 * n2);
    const real_t mes2 = mes * mes;
    const real_t lam_diff = Dxx * mes2 + Dyy * mes2 + Dzz * mes2;
    lam += lam_diff / cell_volume(i);
  }

  if (lam != real_t(0))
    return cfl * cell_volume(i) / lam;
  return TIME_STEP_NO_LIMIT;
}
