#pragma once

// real_t / index_t: the floating-point and integer types shared by every
// manapy_compute_<float bits>_<int bits> package. Each target defines
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS to select them.

#include <cstdint>

#if MANAPY_COMPUTE_FLOAT_BITS == 32
using real_t = float;
#elif MANAPY_COMPUTE_FLOAT_BITS == 64
using real_t = double;
#else
#error "MANAPY_COMPUTE_FLOAT_BITS must be 32 or 64"
#endif

#if MANAPY_COMPUTE_INT_BITS == 32
using index_t = std::int32_t;
#elif MANAPY_COMPUTE_INT_BITS == 64
using index_t = std::int64_t;
#else
#error "MANAPY_COMPUTE_INT_BITS must be 32 or 64"
#endif
