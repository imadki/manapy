// Bindings for count_max_node_cellid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; res is written in
// place.
void count_max_node_cellid_py(CIMat cells, IVec res) {
  count_max_node_cellid(make_view<const index_t, 2>(cells),
                         make_view<index_t, 1>(res));
}

} // namespace

void register_count_max_node_cellid(nb::module_ &m) {
  m.def("count_max_node_cellid", &count_max_node_cellid_py, nb::arg("cells"),
        nb::arg("res").noconvert(),
        "Increments res[node] once for every cell that lists it. Writes "
        "into res in place; size node_cellid's row width off of "
        "max(res) + 1 before calling create_node_cellid.");
}
