#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Which array supplies the far-side cell value when averaging a cell field
// onto a face (translation of celltoface in to_convert.py, which repeats the
// same math across three index lists that differ only in how the far-side
// value is fetched):
//   TwoCell -> u_cell(face_cellid(i, 1))  interior faces (d_innerfaces)
//   Halo    -> u_halo(face_halofid(i))    halo faces (d_halofaces)
//   Ghost   -> u_ghost(i)                 boundary faces (d_boundaryfaces)
enum class CellToFaceKind { TwoCell, Halo, Ghost };

// Cell-to-face averaging for a single face i: u_face(i) is the midpoint
// average of the two cell values straddling the face. Shared verbatim by the
// CPU loops (cpu/celltoface_cpu.cpp) and the CUDA kernels
// (gpu/celltoface_cuda.cu): MANAPY_COMPUTE_HOST_DEVICE compiles it as a plain
// inline function in C++ TUs and as a __host__ __device__ function under
// nvcc, so the exact same arithmetic runs on both host and GPU. Writes
// u_face at index i in place.
template <CellToFaceKind Kind>
MANAPY_COMPUTE_HOST_DEVICE void celltoface_face(
    index_t i, ArrayView<const real_t, 1> u_cell, ArrayView<const real_t, 1> u_ghost,
    ArrayView<const real_t, 1> u_halo, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 1> face_halofid, ArrayView<real_t, 1> u_face) {
  const real_t vv1 = u_cell(face_cellid(i, 0));

  real_t vv2;
  if constexpr (Kind == CellToFaceKind::TwoCell) {
    vv2 = u_cell(face_cellid(i, 1));
  } else if constexpr (Kind == CellToFaceKind::Halo) {
    vv2 = u_halo(face_halofid(i));
  } else {
    vv2 = u_ghost(i);
  }

  u_face(i) = real_t(0.5) * (vv1 + vv2);
}
