#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Which scalar boundary condition sets the ghost value on a boundary face
// (translation of ghost_value_dirichlet / _neumann / _neumannNH / _nonslip in
// to_convert.py, which repeat the same gather over bc_faces and differ only in
// the expression assigned to w_ghost):
//   Dirichlet -> w_ghost(i) = value(i)                       imposed value
//   Neumann   -> w_ghost(i) = w_c(face_cellid(i, 0))         zero normal gradient
//   NeumannNH -> w_ghost(i) = w_c(...) + cst(i) * dist(i)    imposed normal gradient
//   NonSlip   -> w_ghost(i) = -w_c(face_cellid(i, 0))        odd reflection
enum class GhostValueKind { Dirichlet, Neumann, NeumannNH, NonSlip };

// Scalar boundary condition for a single boundary face i. Shared verbatim by
// the CPU loops (cpu/ghost_value_cpu.cpp) and the CUDA kernels
// (gpu/ghost_value_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE compiles it as a plain
// inline function in C++ TUs and as a __host__ __device__ function under nvcc,
// so the exact same arithmetic runs on both host and GPU. Writes w_ghost at
// index i in place.
//
// `w` is the Dirichlet `value` array (indexed by face) for Kind == Dirichlet
// and the cell-centred field `w_c` (indexed by cell) for every other kind --
// they occupy the same slot in the Python originals. cst and face_dist_ortho
// are only read by NeumannNH, face_cellid only by the non-Dirichlet kinds; the
// unused ones are kept in the signature for parity with the Python API.
template <GhostValueKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void ghost_value_face(
    index_t i, ArrayView<const real_t, 1> w, ArrayView<real_t, 1> w_ghost,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const real_t, 1> cst,
    ArrayView<const real_t, 1> face_dist_ortho) {
  if constexpr (Kind == GhostValueKind::Dirichlet) {
    w_ghost(i) = w(i);
  } else {
    const real_t w_cell = w(face_cellid(i, 0));
    if constexpr (Kind == GhostValueKind::Neumann) {
      w_ghost(i) = w_cell;
    } else if constexpr (Kind == GhostValueKind::NeumannNH) {
      w_ghost(i) = w_cell + cst(i) * face_dist_ortho(i);
    } else {
      w_ghost(i) = -w_cell;
    }
  }
}
