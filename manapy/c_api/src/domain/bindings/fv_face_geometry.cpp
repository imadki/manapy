// Bindings for fv_face_geometry. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; fv_coeff,
// fv_corrx/y/z and fv_weight_left are written in place.
void fv_face_geometry_py(CIMat face_cellid, CIVec face_name,
                         CFMat face_normal, CFMat face_center,
                         CIVec face_haloid, CFMat cell_center,
                         CFMat halo_centvol, CFMat cell_shift,
                         FVec fv_coeff, FVec fv_corrx, FVec fv_corry,
                         FVec fv_corrz, FVec fv_weight_left) {
  fv_face_geometry(make_view<const index_t, 2>(face_cellid),
                    make_view<const index_t, 1>(face_name),
                    make_view<const real_t, 2>(face_normal),
                    make_view<const real_t, 2>(face_center),
                    make_view<const index_t, 1>(face_haloid),
                    make_view<const real_t, 2>(cell_center),
                    make_view<const real_t, 2>(halo_centvol),
                    make_view<const real_t, 2>(cell_shift),
                    make_view<real_t, 1>(fv_coeff),
                    make_view<real_t, 1>(fv_corrx),
                    make_view<real_t, 1>(fv_corry),
                    make_view<real_t, 1>(fv_corrz),
                    make_view<real_t, 1>(fv_weight_left));
}

} // namespace

void register_fv_face_geometry(nb::module_ &m) {
  m.def("fv_face_geometry", &fv_face_geometry_py, nb::arg("face_cellid"),
        nb::arg("face_name"), nb::arg("face_normal"), nb::arg("face_center"),
        nb::arg("face_haloid"), nb::arg("cell_center"), nb::arg("halo_centvol"),
        nb::arg("cell_shift"), nb::arg("fv_coeff").noconvert(),
        nb::arg("fv_corrx").noconvert(), nb::arg("fv_corry").noconvert(),
        nb::arg("fv_corrz").noconvert(), nb::arg("fv_weight_left").noconvert(),
        "Per-face coefficients for a finite-volume-style gradient scheme: "
        "fv_coeff scales the normal-direction term, fv_corrx/y/z is the "
        "non-orthogonal correction vector, fv_weight_left is the left-cell "
        "interpolation weight. Raises if the face normal is orthogonal to "
        "the left-to-right direction. Writes into fv_coeff, fv_corrx/y/z "
        "and fv_weight_left in place.");
}
