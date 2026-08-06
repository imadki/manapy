// Bindings for create_cell_cellnid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; cell_cellnid is
// written in place (must already be zeroed).
void create_cell_cellnid_py(CIMat cells, CIMat node_cellid,
                            IMat cell_cellnid) {
  create_cell_cellnid(make_view<const index_t, 2>(cells),
                       make_view<const index_t, 2>(node_cellid),
                       make_view<index_t, 2>(cell_cellnid));
}

} // namespace

void register_create_cell_cellnid(nb::module_ &m) {
  m.def("create_cell_cellnid", &create_cell_cellnid_py, nb::arg("cells"),
        nb::arg("node_cellid"), nb::arg("cell_cellnid").noconvert(),
        "Node-adjacency between cells: for each cell i and each "
        "node-neighboring cell nc, records i into cell_cellnid[nc]. Writes "
        "into cell_cellnid in place; it must already be zeroed and sized "
        "by count_max_cell_cellnid's result.");
}
