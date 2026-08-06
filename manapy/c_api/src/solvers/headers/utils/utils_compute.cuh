#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU launch entry-point declarations for the solver utility kernels (common to
// all solvers). Each launch_<kernel> mirrors the CPU declaration in
// utils_compute.hpp and takes device (CuPy) ArrayViews plus a cudaStream_t.
// (The Gaussian initial-condition kernels are CPU-only and have no entry here.)

// GPU counterpart of update_new_value. One thread per cell; ne_c is updated in
// place on the device.
void launch_update_new_value(ArrayView<real_t, 1> ne_c,
                             ArrayView<const real_t, 1> rez_ne,
                             ArrayView<const real_t, 1> dissip_ne,
                             ArrayView<const real_t, 1> src_ne, real_t dtime,
                             ArrayView<const real_t, 1> cell_volume,
                             cudaStream_t stream);
