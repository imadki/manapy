// _core module entry point for the advection-diffusion (advecdiff) solver,
// shipped as the nested manapy_compute_<float bits>_<int bits>.solvers.advecdiff
// submodule (alongside solvers.advec and solvers.utils). Kernels are ported
// from src/solvers/to_convert.py (see src/solvers/Steps.md). Compiled four
// times, once per precision pair, with MANAPY_COMPUTE_FLOAT_BITS /
// MANAPY_COMPUTE_INT_BITS selecting real_t/index_t.
//
// Batch 1 scaffolds an empty module; register_* calls are added as kernels land.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy advection-diffusion (advecdiff) solver kernels compiled for "
            "float" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
            MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_advecdiff_explicitscheme_convective_2d(m);
  register_advecdiff_explicitscheme_convective_3d(m);
  register_advecdiff_explicitscheme_dissipative(m);
  register_advecdiff_time_step(m);
}
