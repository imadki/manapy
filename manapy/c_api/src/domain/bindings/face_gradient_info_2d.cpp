// Bindings for face_gradient_info_2d. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; face_air_diamond,
// face_param1..4 and face_f1..4 are written in place.
void face_gradient_info_2d_py(CIMat face_cellid, CIMat faces,
                              CIVec face_to_phyid, CFMat ghost_info_flt,
                              CIVec face_name, CFMat face_normal,
                              CFMat cell_center, CFMat halo_centvol,
                              CIVec face_haloid, CFMat nodes,
                              FVec face_air_diamond, FVec face_param1,
                              FVec face_param2, FVec face_param3,
                              FVec face_param4, FMat face_f1, FMat face_f2,
                              FMat face_f3, FMat face_f4, CFMat cell_shift) {
  face_gradient_info_2d(
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 2>(faces),
      make_view<const index_t, 1>(face_to_phyid),
      make_view<const real_t, 2>(ghost_info_flt),
      make_view<const index_t, 1>(face_name),
      make_view<const real_t, 2>(face_normal),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const index_t, 1>(face_haloid),
      make_view<const real_t, 2>(nodes), make_view<real_t, 1>(face_air_diamond),
      make_view<real_t, 1>(face_param1), make_view<real_t, 1>(face_param2),
      make_view<real_t, 1>(face_param3), make_view<real_t, 1>(face_param4),
      make_view<real_t, 2>(face_f1), make_view<real_t, 2>(face_f2),
      make_view<real_t, 2>(face_f3), make_view<real_t, 2>(face_f4),
      make_view<const real_t, 2>(cell_shift));
}

} // namespace

void register_face_gradient_info_2d(nb::module_ &m) {
  m.def("face_gradient_info_2d", &face_gradient_info_2d_py,
        nb::arg("face_cellid"), nb::arg("faces"), nb::arg("face_to_phyid"),
        nb::arg("ghost_info_flt"), nb::arg("face_name"), nb::arg("face_normal"),
        nb::arg("cell_center"), nb::arg("halo_centvol"), nb::arg("face_haloid"),
        nb::arg("nodes"), nb::arg("face_air_diamond").noconvert(),
        nb::arg("face_param1").noconvert(), nb::arg("face_param2").noconvert(),
        nb::arg("face_param3").noconvert(), nb::arg("face_param4").noconvert(),
        nb::arg("face_f1").noconvert(), nb::arg("face_f2").noconvert(),
        nb::arg("face_f3").noconvert(), nb::arg("face_f4").noconvert(),
        nb::arg("cell_shift"),
        "Per-face diamond-scheme geometry (f1..f4, param1..4, air_diamond) "
        "for a Green-Gauss-style gradient at face midpoints on a 2D mesh, "
        "computed from raw mesh/ghost/halo/periodic data. Writes into "
        "face_air_diamond, face_param1..4 and face_f1..4 in place.");
}
