// Bindings for pair_periodic_faces. Compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; face_cellid and
// cell_shift are written in place.
index_t pair_periodic_faces_py(CIVec face_name, CFMat face_center,
                               IMat face_cellid, FMat cell_shift, CFVec cmin,
                               index_t name_lo, index_t name_hi,
                               index_t taxis0, index_t taxis1, index_t saxis,
                               real_t L, real_t dtol) {
  return pair_periodic_faces(
      make_view<const index_t, 1>(face_name),
      make_view<const real_t, 2>(face_center),
      make_view<index_t, 2>(face_cellid), make_view<real_t, 2>(cell_shift),
      make_view<const real_t, 1>(cmin), name_lo, name_hi, taxis0, taxis1,
      saxis, L, dtol);
}

} // namespace

void register_pair_periodic_faces(nb::module_ &m) {
  m.def("pair_periodic_faces", &pair_periodic_faces_py, nb::arg("face_name"),
        nb::arg("face_center"), nb::arg("face_cellid").noconvert(),
        nb::arg("cell_shift").noconvert(), nb::arg("cmin"), nb::arg("name_lo"),
        nb::arg("name_hi"), nb::arg("taxis0"), nb::arg("taxis1"),
        nb::arg("saxis"), nb::arg("L"), nb::arg("dtol"),
        "Same-rank periodic face pairing: matches faces tagged name_lo "
        "(owner shift +L on component saxis) to name_hi (shift -L) by their "
        "transverse coordinate(s) taxis0[,taxis1] (pass taxis1=-1 for a "
        "single transverse axis), and wires face_cellid(.,1) + cell_shift "
        "in place. Returns the number of pairs matched (0 if neither name "
        "is present), -1 if the two sides have different counts, -2 if a "
        "transverse key has no match.");
}
