#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// CPU entry-point declarations for the advection-diffusion (advecdiff) solver,
// ported from src/solvers/to_convert.py (see src/solvers/Steps.md for the batch
// plan). Kernels are added here batch by batch; Batch 1 only scaffolds the
// module.
//
// Convention: advecdiff is a CPU + GPU module, so unless a declaration is
// explicitly commented "CPU-only", it has a matching launch_<kernel> in
// advecdiff_compute.cuh. Mark a kernel CPU-only in its comment here when it has
// no GPU counterpart.

// Explicit convective residual for 2D linear advection (translation of
// _explicitscheme_convective_2d). Zeroes rez_w, then sweeps the interior,
// periodic, halo and boundary face lists, scattering each face's numerical flux
// into rez_w (subtracted from the owner cell, added to the neighbour for
// interior faces). `order` selects MUSCL reconstruction (order 1 = first order,
// no reconstruction/loads). `scheme` selects the numerical flux (see
// common/helpers/numerical_flux.hpp; SCHEME_IDS upwind 0 / centered 1 /
// rusanov 2 / lax_friedrichs 3) and is resolved once by a switch, not per face.
// Unlike the Python original, `scheme` is an explicit argument here (it replaces
// the global _compute_flux binding). face_normal must be >= 3 columns wide.
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

// Explicit CFL time step for advection-diffusion (translation of _time_step):
// min over all cells of cfl * cell_volume / lambda, where lambda sums, over the
// cell's faces, the convective speed |u.n| plus the diffusion term
// (Dxx+Dyy+Dzz) * ||face_normal||^2 / cell_volume. Returns the time step (1e6 if
// no cell has a non-zero contribution). face_normal must be at least 3 columns
// wide. The Python signature's face_measure and dim arguments are unused by the
// computation and omitted.
real_t time_step(ArrayView<const real_t, 1> u, ArrayView<const real_t, 1> v,
                 ArrayView<const real_t, 1> w, real_t cfl,
                 ArrayView<const real_t, 2> face_normal,
                 ArrayView<const real_t, 1> cell_volume,
                 ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                 real_t Dyy, real_t Dzz);

// Diffusion (dissipative) residual (translation of _explicitscheme_dissipative).
// Zeroes dissip_w, then for every face accumulates q = Dxx*wx_face*n0 +
// Dyy*wy_face*n1 + Dzz*wz_face*n2 into the owner cell (+q) and, for interior
// faces (face_name == 0), the neighbour (-q). Dimension-agnostic (one routine
// for 2D and 3D); face_normal must be at least 3 columns wide. dissip_w is
// written in place.
void explicitscheme_dissipative(ArrayView<const real_t, 1> wx_face,
                                ArrayView<const real_t, 1> wy_face,
                                ArrayView<const real_t, 1> wz_face,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const real_t, 2> face_normal,
                                ArrayView<const index_t, 1> face_name,
                                ArrayView<real_t, 1> dissip_w, real_t Dxx,
                                real_t Dyy, real_t Dzz);

// Explicit convective residual for 3D linear advection (translation of
// _explicitscheme_convective_3d). As the 2D version but with the full
// three-component reconstruction and a third periodic axis (face names 55/66
// shift z); w_z / wz_halo are used here.
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
