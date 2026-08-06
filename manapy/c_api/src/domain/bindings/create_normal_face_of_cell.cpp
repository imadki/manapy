// Bindings for create_normal_face_of_cell. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; cell_nf is written
// in place.
void create_normal_face_of_cell_py(CFMat cell_center, CFMat face_center,
                                   CIMat cell_faceid, CFMat face_normal,
                                   FTensor cell_nf) {
  create_normal_face_of_cell(
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(face_center),
      make_view<const index_t, 2>(cell_faceid),
      make_view<const real_t, 2>(face_normal),
      make_view<real_t, 3>(cell_nf));
}

} // namespace

void register_create_normal_face_of_cell(nb::module_ &m) {
  m.def("create_normal_face_of_cell", &create_normal_face_of_cell_py,
        nb::arg("cell_center"), nb::arg("face_center"), nb::arg("cell_faceid"),
        nb::arg("face_normal"), nb::arg("cell_nf").noconvert(),
        "Outward-oriented copy of face_normal for every (cell, local "
        "face) pair: cell_nf(i, j, :) is face_normal(fid, :) flipped if "
        "needed to point away from cell i's center. Writes into cell_nf "
        "in place.");
}
