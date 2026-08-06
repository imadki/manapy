#include "variable_compute.hpp"

#include "common/celltoface_common.hpp"

namespace {

template <CellToFaceKind Kind>
void celltoface_group(ArrayView<const real_t, 1> u_cell, ArrayView<const real_t, 1> u_ghost,
                       ArrayView<const real_t, 1> u_halo,
                       ArrayView<const index_t, 2> face_cellid,
                       ArrayView<const index_t, 1> face_halofid,
                       ArrayView<real_t, 1> u_face,
                       ArrayView<const index_t, 1> face_list) {
  const index_t n = static_cast<index_t>(face_list.size(0));
  for (index_t k = 0; k < n; ++k) {
    const index_t i = face_list(k);
    celltoface_face<Kind>(i, u_cell, u_ghost, u_halo, face_cellid, face_halofid,
                           u_face);
  }
}

} // namespace

void celltoface(
    ArrayView<const real_t, 1> u_cell, ArrayView<real_t, 1> u_face,
    ArrayView<const real_t, 1> u_ghost, ArrayView<const real_t, 1> u_halo,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 1> face_halofid,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_halofaces) {
  celltoface_group<CellToFaceKind::TwoCell>(u_cell, u_ghost, u_halo, face_cellid,
                                             face_halofid, u_face, d_innerfaces);
  celltoface_group<CellToFaceKind::Halo>(u_cell, u_ghost, u_halo, face_cellid,
                                          face_halofid, u_face, d_halofaces);
  celltoface_group<CellToFaceKind::Ghost>(u_cell, u_ghost, u_halo, face_cellid,
                                           face_halofid, u_face, d_boundaryfaces);
}
