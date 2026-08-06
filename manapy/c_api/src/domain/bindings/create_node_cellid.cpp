// Bindings for create_node_cellid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_cellid is
// written in place (must already be zeroed).
void create_node_cellid_py(CIMat cells, IMat node_cellid) {
  create_node_cellid(make_view<const index_t, 2>(cells),
                      make_view<index_t, 2>(node_cellid));
}

} // namespace

void register_create_node_cellid(nb::module_ &m) {
  m.def("create_node_cellid", &create_node_cellid_py, nb::arg("cells"),
        nb::arg("node_cellid").noconvert(),
        "For each node, the sorted (ascending) list of cells that reference "
        "it. Writes into node_cellid in place; it must already be zeroed "
        "and sized by count_max_node_cellid's result.");
}
