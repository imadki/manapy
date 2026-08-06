#pragma once

#include "array_view.hpp"
#include "common/boundary_math.hpp"
#include "precision.hpp"

// Which scalar boundary condition sets the halo-ghost values hanging off a
// halo node (translation of haloghost_value_dirichlet / _neumann / _neumannNH /
// _nonslip in to_convert.py, which repeat the same node -> halo-ghost walk and
// differ only in the expression assigned to w_haloghost):
//   Dirichlet -> w(ghost_id)                          imposed value (see below)
//   Neumann   -> w(ghost_ext_info_int(ghost_id, 0))   zero normal gradient
//   NeumannNH -> w(...) + cst(ghost_id) * 2*|dist|    imposed normal gradient
//   NonSlip   -> -w(ghost_ext_info_int(ghost_id, 0))  odd reflection
enum class HaloGhostValueKind { Dirichlet, Neumann, NeumannNH, NonSlip };

// Scalar boundary condition for every halo ghost attached to halo node i whose
// boundary tag matches BCindex. Shared verbatim by the CPU loops
// (cpu/haloghost_value_cpu.cpp) and the CUDA kernels
// (gpu/haloghost_value_cuda.cu). Writes w_haloghost in place.
//
// `w` is the prescribed per-halo-ghost value array for Kind == Dirichlet and
// the halo *cell* field `w_halo` for every other kind -- they occupy the same
// slot in the Python originals, which is why Dirichlet indexes it by ghost_id
// while the others go through the halo cell in ghost_ext_info_int(ghost_id, 0).
// The two index spaces are different sizes (halos.sizehaloghost vs nbhalos), so
// the distinction matters.
//
// The last column of node_haloghostid holds the number of valid entries in that
// row. ghost_ext_info_int(ghost_id, 0) is the halo cell behind the ghost and
// column 1 its boundary tag; ghost_ext_info_flt(ghost_id, 0) is the signed
// distance whose doubled magnitude NeumannNH uses as the wall distance. cst is
// indexed per ghost, not per node: a ghost is reachable from every node of its
// face, so a per-node constant would make the result depend on which node
// visited it last.
template <HaloGhostValueKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void haloghost_value_node(
    index_t i, ArrayView<const real_t, 1> w, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const real_t, 1> cst) {
  const index_t count = node_haloghostid(i, node_haloghostid.size(1) - 1);
  for (index_t j = 0; j < count; ++j) {
    const index_t ghost_id = node_haloghostid(i, j);
    if (ghost_ext_info_int(ghost_id, 1) != BCindex)
      continue;

    if constexpr (Kind == HaloGhostValueKind::Dirichlet) {
      w_haloghost(ghost_id) = w(ghost_id);
    } else {
      const real_t w_cell = w(ghost_ext_info_int(ghost_id, 0));
      if constexpr (Kind == HaloGhostValueKind::Neumann) {
        w_haloghost(ghost_id) = w_cell;
      } else if constexpr (Kind == HaloGhostValueKind::NonSlip) {
        w_haloghost(ghost_id) = -w_cell;
      } else {
        const real_t dist =
            real_t(2) * boundary_abs(ghost_ext_info_flt(ghost_id, 0));
        w_haloghost(ghost_id) = w_cell + cst(ghost_id) * dist;
      }
    }
  }
}
