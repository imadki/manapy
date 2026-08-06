#pragma once

#include <cmath>

#include "array_view.hpp" // MANAPY_COMPUTE_HOST_DEVICE
#include "precision.hpp"

// Portable scalar math shared by the boundary-condition kernels. On the device
// pass nvcc picks the CUDA intrinsic; on the host pass these are the <cmath>
// versions, so the exact same routine compiles for both.

MANAPY_COMPUTE_HOST_DEVICE
real_t boundary_sqrt(real_t a) {
#if defined(__CUDA_ARCH__)
  return sqrt(a);
#else
  return std::sqrt(a);
#endif
}

MANAPY_COMPUTE_HOST_DEVICE
real_t boundary_abs(real_t a) {
#if defined(__CUDA_ARCH__)
  return fabs(a);
#else
  return std::fabs(a);
#endif
}
