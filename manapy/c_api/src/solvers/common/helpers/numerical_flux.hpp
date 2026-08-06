#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Solver-agnostic numerical flux for linear scalar advection, shared across the
// solvers (e.g. advecdiff). The scheme is a compile-time template parameter so
// the flux inlines into the convective kernel with no per-face branch and no
// indirect call: a kernel is instantiated once per FluxScheme and the host
// launcher does a single switch(scheme) per launch to pick the instantiation.
// This is the C++ equivalent of the Python setup(dim, scheme) that binds one
// _compute_flux body, but resolved by the compiler instead of a function
// pointer (function pointers are slow on the GPU).
//
// The Python side maps scheme names to the ids the launcher switches on with:
//   SCHEME_IDS = {"upwind": 0, "centered": 1, "rusanov": 2, "lax_friedrichs": 3}
// Rusanov and LaxFriedrichs are identical for linear scalar advection and share
// the same branch below (kept as distinct enum values for API parity).
enum class FluxScheme { Upwind, Centered, Rusanov, LaxFriedrichs };

// Numerical flux through one face given the left/right states, the advection
// velocity (u_face, v_face, w_face) at the face and its normal (read at indices
// 0, 1 and 2, so at least 3 components; z is 0 in 2D). Returns the flux; the
// caller scatters it into the residual. MANAPY_COMPUTE_HOST_DEVICE, so the same
// arithmetic runs on host and GPU.
template <FluxScheme S>
MANAPY_COMPUTE_HOST_DEVICE real_t
numerical_flux(real_t w_l, real_t w_r, real_t u_face, real_t v_face,
               real_t w_face, ArrayView<const real_t, 1> normal) {
  const real_t sign =
      u_face * normal(0) + v_face * normal(1) + w_face * normal(2);

  if constexpr (S == FluxScheme::Upwind) {
    return sign >= real_t(0) ? sign * w_l : sign * w_r;
  } else if constexpr (S == FluxScheme::Centered) {
    return sign * real_t(0.5) * (w_l + w_r);
  } else { // Rusanov / LaxFriedrichs -- identical for linear scalar advection
    const real_t abs_sign = sign < real_t(0) ? -sign : sign;
    return real_t(0.5) * sign * (w_l + w_r) -
           real_t(0.5) * abs_sign * (w_r - w_l);
  }
}
