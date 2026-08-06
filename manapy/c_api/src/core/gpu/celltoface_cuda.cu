#include <algorithm>

#include "common/celltoface_common.hpp"
#include "variable_compute.cuh"

namespace {

// One thread per entry in face_list; a grid-stride loop covers lists larger
// than one grid. ArrayViews are passed by value (their data pointers already
// reference device memory) and each thread reuses the shared host/device
// per-face routine, templated on which array supplies the far-side value.
template <CellToFaceKind Kind>
__global__ void celltoface_kernel(ArrayView<const real_t, 1> u_cell,
                                   ArrayView<const real_t, 1> u_ghost,
                                   ArrayView<const real_t, 1> u_halo,
                                   ArrayView<const index_t, 2> face_cellid,
                                   ArrayView<const index_t, 1> face_halofid,
                                   ArrayView<real_t, 1> u_face,
                                   ArrayView<const index_t, 1> face_list,
                                   index_t n) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < n; k += stride) {
    const index_t i = face_list(k);
    celltoface_face<Kind>(i, u_cell, u_ghost, u_halo, face_cellid, face_halofid,
                           u_face);
  }
}

template <CellToFaceKind Kind>
void launch_celltoface_group(ArrayView<const real_t, 1> u_cell,
                              ArrayView<const real_t, 1> u_ghost,
                              ArrayView<const real_t, 1> u_halo,
                              ArrayView<const index_t, 2> face_cellid,
                              ArrayView<const index_t, 1> face_halofid,
                              ArrayView<real_t, 1> u_face,
                              ArrayView<const index_t, 1> face_list,
                              cudaStream_t stream) {
  const index_t n = static_cast<index_t>(face_list.size(0));
  if (n <= 0)
    return;

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
  celltoface_kernel<Kind><<<blocks, threads, 0, stream>>>(
      u_cell, u_ghost, u_halo, face_cellid, face_halofid, u_face, face_list, n);
}

} // namespace

void launch_celltoface(
    ArrayView<const real_t, 1> u_cell, ArrayView<real_t, 1> u_face,
    ArrayView<const real_t, 1> u_ghost, ArrayView<const real_t, 1> u_halo,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 1> face_halofid,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_halofaces, cudaStream_t stream) {
  launch_celltoface_group<CellToFaceKind::TwoCell>(
      u_cell, u_ghost, u_halo, face_cellid, face_halofid, u_face, d_innerfaces,
      stream);
  launch_celltoface_group<CellToFaceKind::Halo>(
      u_cell, u_ghost, u_halo, face_cellid, face_halofid, u_face, d_halofaces,
      stream);
  launch_celltoface_group<CellToFaceKind::Ghost>(
      u_cell, u_ghost, u_halo, face_cellid, face_halofid, u_face,
      d_boundaryfaces, stream);
}
