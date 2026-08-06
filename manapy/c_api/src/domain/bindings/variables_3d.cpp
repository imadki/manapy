// Bindings for variables_3d. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_R_x/y/z,
// node_lambda_x/y/z and node_number are written in place (node_R_x/y/z and
// node_number are accumulators and must start zeroed).
void variables_3d_py(CFMat cell_center, CIMat node_cellid, CIMat node_haloid,
                     CIMat node_ghostid, CIMat node_haloghostid,
                     CIMat node_periodicid, CFMat nodes, CIVec node_oldname,
                     CFMat ghost_info_flt, CFMat ext_ghost_info_flt,
                     CFMat halo_centvol, FVec node_R_x, FVec node_R_y,
                     FVec node_R_z, FVec node_lambda_x, FVec node_lambda_y,
                     FVec node_lambda_z, IVec node_number,
                     CFMat cell_shift) {
  variables_3d(make_view<const real_t, 2>(cell_center),
               make_view<const index_t, 2>(node_cellid),
               make_view<const index_t, 2>(node_haloid),
               make_view<const index_t, 2>(node_ghostid),
               make_view<const index_t, 2>(node_haloghostid),
               make_view<const index_t, 2>(node_periodicid),
               make_view<const real_t, 2>(nodes),
               make_view<const index_t, 1>(node_oldname),
               make_view<const real_t, 2>(ghost_info_flt),
               make_view<const real_t, 2>(ext_ghost_info_flt),
               make_view<const real_t, 2>(halo_centvol),
               make_view<real_t, 1>(node_R_x), make_view<real_t, 1>(node_R_y),
               make_view<real_t, 1>(node_R_z),
               make_view<real_t, 1>(node_lambda_x),
               make_view<real_t, 1>(node_lambda_y),
               make_view<real_t, 1>(node_lambda_z),
               make_view<index_t, 1>(node_number),
               make_view<const real_t, 2>(cell_shift));
}

} // namespace

void register_variables_3d(nb::module_ &m) {
  m.def("variables_3d", &variables_3d_py, nb::arg("cell_center"),
        nb::arg("node_cellid"), nb::arg("node_haloid"), nb::arg("node_ghostid"),
        nb::arg("node_haloghostid"), nb::arg("node_periodicid"),
        nb::arg("nodes"), nb::arg("node_oldname"), nb::arg("ghost_info_flt"),
        nb::arg("ext_ghost_info_flt"), nb::arg("halo_centvol"),
        nb::arg("node_R_x").noconvert(), nb::arg("node_R_y").noconvert(),
        nb::arg("node_R_z").noconvert(), nb::arg("node_lambda_x").noconvert(),
        nb::arg("node_lambda_y").noconvert(),
        nb::arg("node_lambda_z").noconvert(),
        nb::arg("node_number").noconvert(), nb::arg("cell_shift"),
        "3D counterpart of variables_2d: solves a 3x3 moment system for "
        "node_lambda_x/y/z via the closed-form cofactor/adjugate "
        "expressions. Raises if the moment matrix is singular. Writes into "
        "node_R_x/y/z (accumulators, must start zeroed), node_lambda_x/y/z "
        "and node_number (accumulator, must start zeroed) in place.");
}
