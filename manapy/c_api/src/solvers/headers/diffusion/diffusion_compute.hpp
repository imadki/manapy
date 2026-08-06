#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// CPU entry-point declarations for the pure-diffusion (diffusion) solver,
// ported from src/solvers/to_convert.py. Unlike advecdiff there is no
// convective residual here: the solver is the dissipative flux plus its own
// (diffusion-only) CFL time step. The forward-Euler update is not duplicated
// here -- it is the shared solvers.utils.update_new_value.
//
// Convention: diffusion is a CPU + GPU module, so unless a declaration is
// explicitly commented "CPU-only", it has a matching launch_<kernel> in
// diffusion_compute.cuh.

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

// Explicit CFL time step for pure diffusion (translation of _time_step): min
// over all cells of cfl * cell_volume / lambda, where lambda sums, over the
// cell's faces, only the diffusion term (Dxx+Dyy+Dzz) * ||face_normal||^2 /
// cell_volume -- there is no convective |u.n| contribution here. Returns the
// time step (1e6 if no cell has a non-zero contribution). face_normal must be
// at least 3 columns wide. The Python signature's u, v, w, face_measure and dim
// arguments are unused by the computation and omitted (the binding still takes
// them, for signature parity).
real_t time_step(real_t cfl, ArrayView<const real_t, 2> face_normal,
                 ArrayView<const real_t, 1> cell_volume,
                 ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                 real_t Dyy, real_t Dzz);
