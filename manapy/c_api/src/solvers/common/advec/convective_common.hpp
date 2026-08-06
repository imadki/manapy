#pragma once

// The per-cell residual scatter uses scatter_add (host += / device atomicAdd).
#include "common/helpers/scatter.hpp"

// Shared building blocks for the explicit convective residual kernels (2D and
// 3D). Both sweep four disjoint face lists -- interior, periodic-boundary,
// halo and physical boundary faces -- that share the same reconstruction +
// numerical flux but differ in where the far-side state comes from and how the
// flux is scattered back into the per-cell residual rez_w (via scatter_add,
// common/helpers/scatter.hpp).
enum class ConvectiveFaceKind { Inner, Periodic, Halo, Boundary };
