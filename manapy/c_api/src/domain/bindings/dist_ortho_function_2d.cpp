// Bindings for dist_ortho_function_2d. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; face_dist_ortho is
// written in place, only at the indices gathered by d_innerfaces /
// d_boundaryfaces.
void dist_ortho_function_2d_py(CIVec d_innerfaces, CIVec d_boundaryfaces,
                               CIMat face_cellid, CFMat cell_center,
                               CFMat face_center, CFMat face_normal,
                               FVec face_dist_ortho) {
  dist_ortho_function_2d(make_view<const index_t, 1>(d_innerfaces),
                          make_view<const index_t, 1>(d_boundaryfaces),
                          make_view<const index_t, 2>(face_cellid),
                          make_view<const real_t, 2>(cell_center),
                          make_view<const real_t, 2>(face_center),
                          make_view<const real_t, 2>(face_normal),
                          make_view<real_t, 1>(face_dist_ortho));
}

} // namespace

void register_dist_ortho_function_2d(nb::module_ &m) {
  m.def("dist_ortho_function_2d", &dist_ortho_function_2d_py,
        nb::arg("d_innerfaces"), nb::arg("d_boundaryfaces"),
        nb::arg("face_cellid"), nb::arg("cell_center"), nb::arg("face_center"),
        nb::arg("face_normal"), nb::arg("face_dist_ortho").noconvert(),
        "Per-face orthogonal distance used by some FV diffusion schemes: "
        "twice the projection distance for a boundary face, the sum of "
        "both sides' projection distances for an interior face. "
        "d_innerfaces/d_boundaryfaces are gather lists of face indices. "
        "Writes into face_dist_ortho in place, only at those indices.");
}
