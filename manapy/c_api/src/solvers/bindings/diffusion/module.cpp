// _core module entry point for the pure-diffusion (diffusion) solver, shipped
// as the nested manapy_compute_<float bits>_<int bits>.solvers.diffusion
// submodule (alongside solvers.advec, solvers.advecdiff and solvers.utils).
// Kernels are ported from src/solvers/to_convert.py. Compiled four times, once
// per precision pair, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS
// selecting real_t/index_t.
//
// The solver's forward-Euler update is not registered here: it is identical
// across solvers and lives in solvers.utils.update_new_value.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy pure-diffusion solver kernels compiled for "
            "float" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
            MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_diffusion_explicitscheme_dissipative(m);
  register_diffusion_time_step(m);
}
