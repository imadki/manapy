#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// CPU entry points for the advection (advec) solver kernels. Each function
// loops over the relevant mesh elements on the host and calls the shared
// per-element/per-face routine in common/advec/. The GPU launch counterparts
// are declared in advec_compute.cuh.

// Explicit finite-volume convective residual for 2D linear advection
// (translation of _explicitscheme_convective_2d). Zeroes rez_w, then sweeps the
// interior, periodic-boundary, halo and physical-boundary face lists,
// scattering each face's numerical flux into the per-cell residual rez_w
// (subtracted from the owner cell, added to the neighbour for interior faces).
// MUSCL reconstruction is limited by psi and disabled for order == 1; scheme
// selects the numerical flux (see common/advec/numerical_flux_common.hpp).
// face_normal must be at least 3 columns wide. w_z / wz_halo from the Python
// signature are unused in 2D and omitted here.
void explicitscheme_convective_2d(
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
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme);

// Explicit finite-volume convective residual for 3D linear advection
// (translation of _explicitscheme_convective_3d). Same structure as the 2D
// version above, but reconstructs with the full three-component gradient and
// handles a third periodic axis (face names 55/66 shift z). w_z / wz_halo are
// used here. face_normal must be at least 3 columns wide.
void explicitscheme_convective_3d(
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
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme);

// Explicit CFL time step (translation of _time_step): min over all cells of
// cfl * cell_volume / lambda, where lambda is the sum over the cell's faces of
// |u.n|. Returns the time step (1e6 if no cell has a non-zero convective
// speed). face_normal must be at least 3 columns wide. The Python signature's
// face_measure and dim arguments are unused by the computation and omitted.
real_t time_step(ArrayView<const real_t, 1> u, ArrayView<const real_t, 1> v,
                 ArrayView<const real_t, 1> w, real_t cfl,
                 ArrayView<const real_t, 2> face_normal,
                 ArrayView<const real_t, 1> cell_volume,
                 ArrayView<const index_t, 2> cell_faceid);

// update_new_value moved to solvers.utils (common to all solvers); see
// headers/utils/utils_compute.hpp.
