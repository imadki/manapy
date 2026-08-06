#pragma once

// The per-cell residual scatter uses scatter_add (host += / device atomicAdd),
// and the flux is the compile-time-dispatched numerical_flux<FluxScheme>.
#include "common/helpers/numerical_flux.hpp"
#include "common/helpers/scatter.hpp"

// Shared building blocks for the advecdiff explicit convective residual kernels
// (2D and 3D). Like the advec solver, both sweep four disjoint face lists --
// interior, periodic-boundary, halo and physical boundary faces -- that share
// the same reconstruction + numerical flux but differ in where the far-side
// state comes from and how the flux is scattered into the residual rez_w.
//
// advecdiff keeps its own copy of this enum (rather than including advec's) so
// the two solvers stay decoupled; scatter_add and numerical_flux are the truly
// shared pieces and live in common/helpers/.
enum class ConvectiveFaceKind { Inner, Periodic, Halo, Boundary };
