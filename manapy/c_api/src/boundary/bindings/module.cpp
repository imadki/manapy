// _core module entry point for the boundary-condition kernels, shipped as the
// manapy_compute_<float bits>_<int bits>.boundary submodule. Defines NB_MODULE
// once and delegates to each kernel's register_* function
// (bindings/registry.hpp), keeping the kernels in separate translation units.
// Compiled four times, once per precision pair, with MANAPY_COMPUTE_FLOAT_BITS /
// MANAPY_COMPUTE_INT_BITS selecting real_t/index_t.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy boundary-condition kernels compiled for float" MANAPY_COMPUTE_STR(
      MANAPY_COMPUTE_FLOAT_BITS) " data and int" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS)
      " indices";

  register_ghost_value(m);
  register_haloghost_value(m);
  register_ghost_value_slip_2d(m);
  register_ghost_value_slip_3d(m);
  register_haloghost_value_slip_2d(m);
  register_haloghost_value_slip_3d(m);
}
