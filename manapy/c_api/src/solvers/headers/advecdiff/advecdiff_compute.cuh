#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU launch entry-point declarations for the advection-diffusion (advecdiff)
// solver. Each launch_<kernel> mirrors the CPU declaration in
// advecdiff_compute.hpp and takes device (CuPy) ArrayViews plus a cudaStream_t.

// GPU counterpart of explicitscheme_convective_2d. Zeroes rez_w on the device,
// resolves `scheme` with a single switch, then launches one thread-per-face
// kernel per face list (atomicAdd scatter). See advecdiff_compute.hpp for the
// argument semantics.
void launch_explicitscheme_convective_2d(
    ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
    ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> wx_halo,
    ArrayView<const real_t, 1> wy_halo, ArrayView<const real_t, 1> psi,
    ArrayView<const real_t, 1> psi_halo, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_periodicboundaryfaces,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme,
    cudaStream_t stream);

// GPU counterpart of explicitscheme_dissipative. One thread per face; dissip_w
// is zeroed then scattered into with atomicAdd, in place on the device.
void launch_explicitscheme_dissipative(
    ArrayView<const real_t, 1> wx_face, ArrayView<const real_t, 1> wy_face,
    ArrayView<const real_t, 1> wz_face,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_name, ArrayView<real_t, 1> dissip_w,
    real_t Dxx, real_t Dyy, real_t Dzz, cudaStream_t stream);

// GPU counterpart of time_step. Reduces the per-cell CFL candidates (convective
// + diffusion) with an atomicMin on the device and returns the resulting time
// step (the stream is synchronised before the value is read back).
real_t launch_time_step(ArrayView<const real_t, 1> u,
                        ArrayView<const real_t, 1> v,
                        ArrayView<const real_t, 1> w, real_t cfl,
                        ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 1> cell_volume,
                        ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                        real_t Dyy, real_t Dzz, cudaStream_t stream);

// GPU counterpart of explicitscheme_convective_3d.
void launch_explicitscheme_convective_3d(
    ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
    ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<const real_t, 1> wx_halo, ArrayView<const real_t, 1> wy_halo,
    ArrayView<const real_t, 1> wz_halo, ArrayView<const real_t, 1> psi,
    ArrayView<const real_t, 1> psi_halo, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_periodicboundaryfaces,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme,
    cudaStream_t stream);
