// Bindings for create_bf_cellid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; intersect and
// bf_cellid are written in place (intersect is scratch, sized 2).
void create_bf_cellid_py(CIMat phy_faces, CIMat node_cellid,
                         CIVec phyid_to_faceid, CIMat cell_faceid,
                         IVec intersect, IMat bf_cellid) {
  create_bf_cellid(make_view<const index_t, 2>(phy_faces),
                    make_view<const index_t, 2>(node_cellid),
                    make_view<const index_t, 1>(phyid_to_faceid),
                    make_view<const index_t, 2>(cell_faceid),
                    make_view<index_t, 1>(intersect),
                    make_view<index_t, 2>(bf_cellid));
}

} // namespace

void register_create_bf_cellid(nb::module_ &m) {
  m.def("create_bf_cellid", &create_bf_cellid_py, nb::arg("phy_faces"),
        nb::arg("node_cellid"), nb::arg("phyid_to_faceid"),
        nb::arg("cell_faceid"), nb::arg("intersect").noconvert(),
        nb::arg("bf_cellid").noconvert(),
        "For each physical (boundary) face, the local cell it belongs to "
        "and that cell's local face index. Writes into bf_cellid in place "
        "(intersect is scratch, sized 2). Raises if a physical face can't "
        "be resolved to a cell/face.");
}
