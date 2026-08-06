#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU launch entry-point declarations for the pure-diffusion (diffusion)
// solver. Each launch_<kernel> mirrors the CPU declaration in
// diffusion_compute.hpp and takes device (CuPy) ArrayViews plus a cudaStream_t.

// GPU counterpart of explicitscheme_dissipative. One thread per face; dissip_w
// is zeroed then scattered into with atomicAdd, in place on the device.
void launch_explicitscheme_dissipative(
    ArrayView<const real_t, 1> wx_face, ArrayView<const real_t, 1> wy_face,
    ArrayView<const real_t, 1> wz_face,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_name, ArrayView<real_t, 1> dissip_w,
    real_t Dxx, real_t Dyy, real_t Dzz, cudaStream_t stream);

// GPU counterpart of time_step. Reduces the per-cell (diffusion-only) CFL
// candidates with an atomicMin on the device and returns the resulting time
// step (the stream is synchronised before the value is read back).
real_t launch_time_step(real_t cfl, ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 1> cell_volume,
                        ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                        real_t Dyy, real_t Dzz, cudaStream_t stream);
