// Bindings for create_ghost_info. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; ghost_info_int and
// ghost_info_flt are written in place.
void create_ghost_info_py(CIMat bf_cellid, CFMat cell_center,
                          CIMat cell_faceid, CIVec cell_loctoglob,
                          CIMat faces, CFMat nodes, CIVec face_oldname,
                          CFMat face_normal, CFMat face_center,
                          CFVec face_measure, IMat ghost_info_int,
                          FMat ghost_info_flt, index_t dim) {
  create_ghost_info(make_view<const index_t, 2>(bf_cellid),
                     make_view<const real_t, 2>(cell_center),
                     make_view<const index_t, 2>(cell_faceid),
                     make_view<const index_t, 1>(cell_loctoglob),
                     make_view<const index_t, 2>(faces),
                     make_view<const real_t, 2>(nodes),
                     make_view<const index_t, 1>(face_oldname),
                     make_view<const real_t, 2>(face_normal),
                     make_view<const real_t, 2>(face_center),
                     make_view<const real_t, 1>(face_measure),
                     make_view<index_t, 2>(ghost_info_int),
                     make_view<real_t, 2>(ghost_info_flt), dim);
}

} // namespace

void register_create_ghost_info(nb::module_ &m) {
  m.def("create_ghost_info", &create_ghost_info_py, nb::arg("bf_cellid"),
        nb::arg("cell_center"), nb::arg("cell_faceid"),
        nb::arg("cell_loctoglob"), nb::arg("faces"), nb::arg("nodes"),
        nb::arg("face_oldname"), nb::arg("face_normal"), nb::arg("face_center"),
        nb::arg("face_measure"), nb::arg("ghost_info_int").noconvert(),
        nb::arg("ghost_info_flt").noconvert(), nb::arg("dim"),
        "Per boundary cell (from create_bf_cellid), the mirrored ghost "
        "cell center reflected across its boundary face, plus a gamma "
        "weight, face center/normal, old name and global id. dim (2 or 3) "
        "selects the gamma formula. Writes into ghost_info_int and "
        "ghost_info_flt in place.");
}
