// _core module entry point for the advection (advec) solver, shipped as the
// nested manapy_compute_<float bits>_<int bits>.solvers.advec submodule
// (alongside solvers.utils). Kernels are ported from the manapy advection
// solver. Compiled four times, once per precision pair, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting real_t/index_t.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy advection (advec) solver kernels compiled for float"
      MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
      MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_explicitscheme_convective_2d(m);
  register_explicitscheme_convective_3d(m);
  register_time_step(m);
}
