// _core module entry point for the solver utilities, shipped as the
// manapy_compute_<float bits>_<int bits>.solvers.utils submodule: kernels
// common to all solvers (advec, advecDiff, ...) rather than to any one of them.
// Mixed CPU + GPU (per-function; the Gaussian init kernels are CPU-only,
// update_new_value has a GPU counterpart). Compiled four times, once per
// precision pair, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS
// selecting real_t/index_t.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy solver utility kernels (common to all solvers) compiled "
            "for float" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
            MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_initialisation_gaussian(m);
  register_update_new_value(m);
}
