#pragma once

#include "manapy_compute_types.hpp"

// Each kernel's binding TU exposes one register_* function that adds its Python
// entry points to the module. bindings/advec/module.cpp defines
// NB_MODULE(_core) and calls them all, so kernels stay in separate translation
// units instead of one growing module file.
void register_explicitscheme_convective_2d(nb::module_ &m);
void register_explicitscheme_convective_3d(nb::module_ &m);
void register_time_step(nb::module_ &m);

// solvers.utils submodule (kernels common to all solvers). Mixed CPU + GPU:
// initialisation_gaussian is CPU-only, update_new_value has a GPU counterpart.
void register_initialisation_gaussian(nb::module_ &m);
void register_update_new_value(nb::module_ &m);

// solvers.advecdiff submodule (advection-diffusion). Prefixed to avoid clashing
// with the identically-named advec kernels in this shared registry.
void register_advecdiff_explicitscheme_convective_2d(nb::module_ &m);
void register_advecdiff_explicitscheme_convective_3d(nb::module_ &m);
void register_advecdiff_explicitscheme_dissipative(nb::module_ &m);
void register_advecdiff_time_step(nb::module_ &m);

// solvers.diffusion submodule (pure diffusion: dissipative flux + its own
// diffusion-only CFL time step; the forward-Euler update comes from
// solvers.utils). Prefixed like advecdiff's, for the same reason.
void register_diffusion_explicitscheme_dissipative(nb::module_ &m);
void register_diffusion_time_step(nb::module_ &m);
