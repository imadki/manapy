#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Solver-agnostic scatter helpers shared across the manapy solvers (advec,
// advecDiff, ...). Kept out of any one solver's common/ directory so every
// solver can include them.

// Accumulate `val` into a(idx). When a scatter is driven from an element loop
// (e.g. one GPU thread per face writing into a per-cell array), several threads
// race to update the same slot, so on the device we use atomicAdd; on the host
// the loop is serial and a plain += is correct and faster. __CUDA_ARCH__ is
// defined only while nvcc compiles the device pass.
MANAPY_COMPUTE_HOST_DEVICE
void scatter_add(ArrayView<real_t, 1> a, index_t idx, real_t val) {
#if defined(__CUDA_ARCH__)
  atomicAdd(&a(idx), val);
#else
  a(idx) += val;
#endif
}
