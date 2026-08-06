// Bindings for variables_2d. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_R_x/y,
// node_lambda_x/y and node_number are written in place (node_R_x/y and
// node_number are accumulators and must start zeroed).
void variables_2d_py(CFMat cell_center, CIMat node_cellid, CIMat node_haloid,
                     CIMat node_ghostid, CIMat node_haloghostid,
                     CIMat node_periodicid, CFMat nodes, CIVec node_oldname,
                     CFMat ghost_info_flt, CFMat ext_ghost_info_flt,
                     CFMat halo_centvol, FVec node_R_x, FVec node_R_y,
                     FVec node_lambda_x, FVec node_lambda_y,
                     IVec node_number, CFMat cell_shift) {
  variables_2d(make_view<const real_t, 2>(cell_center),
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
               make_view<real_t, 1>(node_lambda_x),
               make_view<real_t, 1>(node_lambda_y),
               make_view<index_t, 1>(node_number),
               make_view<const real_t, 2>(cell_shift));
}

} // namespace

void register_variables_2d(nb::module_ &m) {
  m.def("variables_2d", &variables_2d_py, nb::arg("cell_center"),
        nb::arg("node_cellid"), nb::arg("node_haloid"), nb::arg("node_ghostid"),
        nb::arg("node_haloghostid"), nb::arg("node_periodicid"),
        nb::arg("nodes"), nb::arg("node_oldname"), nb::arg("ghost_info_flt"),
        nb::arg("ext_ghost_info_flt"), nb::arg("halo_centvol"),
        nb::arg("node_R_x").noconvert(), nb::arg("node_R_y").noconvert(),
        nb::arg("node_lambda_x").noconvert(),
        nb::arg("node_lambda_y").noconvert(),
        nb::arg("node_number").noconvert(), nb::arg("cell_shift"),
        "Per-node least-squares gradient-interpolation weights "
        "(node_lambda_x/y), solved from a 2x2 moment matrix accumulated "
        "over every neighboring cell/ghost/periodic-image/halo-ghost/halo "
        "center. Raises if the moment matrix is singular. Writes into "
        "node_R_x/y (accumulators, must start zeroed), node_lambda_x/y and "
        "node_number (accumulator, must start zeroed) in place.");
}
