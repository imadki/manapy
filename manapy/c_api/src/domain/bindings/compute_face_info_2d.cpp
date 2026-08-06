// Bindings for compute_face_info_2d. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; face_measure,
// face_center and face_normal are written in place.
void compute_face_info_2d_py(CIMat faces, CFMat nodes, CIMat face_cellid,
                             CFMat cell_center, FVec face_measure,
                             FMat face_center, FMat face_normal) {
  compute_face_info_2d(make_view<const index_t, 2>(faces),
                        make_view<const real_t, 2>(nodes),
                        make_view<const index_t, 2>(face_cellid),
                        make_view<const real_t, 2>(cell_center),
                        make_view<real_t, 1>(face_measure),
                        make_view<real_t, 2>(face_center),
                        make_view<real_t, 2>(face_normal));
}

} // namespace

void register_compute_face_info_2d(nb::module_ &m) {
  m.def("compute_face_info_2d", &compute_face_info_2d_py, nb::arg("faces"),
        nb::arg("nodes"), nb::arg("face_cellid"), nb::arg("cell_center"),
        nb::arg("face_measure").noconvert(), nb::arg("face_center").noconvert(),
        nb::arg("face_normal").noconvert(),
        "Measure (length), center and outward normal of every 2D face (an "
        "edge between two nodes); normal points away from face_cellid(i, "
        "0). Writes into face_measure, face_center and face_normal in "
        "place.");
}
