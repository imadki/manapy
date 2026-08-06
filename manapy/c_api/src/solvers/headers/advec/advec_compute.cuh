#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU launch entry points for the advection (advec) solver kernels. Each takes
// device (CuPy) ArrayViews and enqueues its kernels on `stream`. The CPU
// counterparts are declared in advec_compute.hpp.

// GPU counterpart of explicitscheme_convective_2d. Zeroes rez_w on the device,
// then launches one thread-per-face kernel per face list (interior, periodic,
// halo, boundary); the scatter into rez_w uses atomicAdd since several faces
// update the same cell. See advec_compute.hpp for the argument semantics.
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

// GPU counterpart of explicitscheme_convective_3d. See advec_compute.hpp for
// the argument semantics; same launch strategy as the 2D version.
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

// GPU counterpart of time_step. Reduces the per-cell CFL candidates with an
// atomicMin on the device and returns the resulting time step (the stream is
// synchronised before the value is read back).
real_t launch_time_step(ArrayView<const real_t, 1> u,
                        ArrayView<const real_t, 1> v,
                        ArrayView<const real_t, 1> w, real_t cfl,
                        ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 1> cell_volume,
                        ArrayView<const index_t, 2> cell_faceid,
                        cudaStream_t stream);

// launch_update_new_value moved to solvers.utils (common to all solvers); see
// headers/utils/utils_compute.cuh.
