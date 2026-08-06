#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Numerical flux at a single face for linear advection, shared verbatim by the
// CPU loop and the CUDA kernel of the advection solver: MANAPY_COMPUTE_HOST_DEVICE
// compiles it as a plain inline function in C++ TUs and as a __host__ __device__
// function under nvcc, so the same arithmetic runs on host and GPU. This is an
// internal helper (translation of _numerical_flux_body), never bound to Python.
//
// `scheme` selects the flux: 0 = upwind, 1 = centered, anything else = Rusanov
// / local Lax-Friedrichs (which equals upwind for linear advection). `w_l` /
// `w_r` are the left/right cell states, (u_face, v_face, w_face) the advection
// velocity at the face and `normal` its (outward) normal. The result is written
// to flux_w[0] in place.
//
// The Python side maps scheme names to these codes with:
//   SCHEME_IDS = {"upwind": 0, "centered": 1, "rusanov": 2, "lax_friedrichs": 2}

// std::fabs is avoided so the routine stays usable from both host and device
// code without pulling in <cmath>.
MANAPY_COMPUTE_HOST_DEVICE
real_t numerical_flux_abs(real_t a) { return a < real_t(0) ? -a : a; }

// Scalar core: returns the flux directly. Used by the convective kernels, which
// need the value rather than an out-parameter buffer. `normal` is read at
// indices 0, 1 and 2, so it must have at least three components (z is 0 in 2D).
MANAPY_COMPUTE_HOST_DEVICE
real_t numerical_flux(index_t scheme, real_t w_l, real_t w_r, real_t u_face,
                      real_t v_face, real_t w_face,
                      ArrayView<const real_t, 1> normal) {
  const real_t sign =
      u_face * normal(0) + v_face * normal(1) + w_face * normal(2);

  if (scheme == index_t(0)) { // upwind
    return sign >= real_t(0) ? sign * w_l : sign * w_r;
  } else if (scheme == index_t(1)) { // centered
    return sign * real_t(0.5) * (w_l + w_r);
  } else { // rusanov / local Lax-Friedrichs (== upwind for linear advection)
    return real_t(0.5) * sign * (w_l + w_r) -
           real_t(0.5) * numerical_flux_abs(sign) * (w_r - w_l);
  }
}

// Out-parameter wrapper, mirroring the Python _numerical_flux_body signature:
// writes the flux into flux_w[0] in place.
MANAPY_COMPUTE_HOST_DEVICE
void numerical_flux_body(index_t scheme, real_t w_l, real_t w_r, real_t u_face,
                         real_t v_face, real_t w_face,
                         ArrayView<const real_t, 1> normal,
                         ArrayView<real_t, 1> flux_w) {
  flux_w(0) =
      numerical_flux(scheme, w_l, w_r, u_face, v_face, w_face, normal);
}
