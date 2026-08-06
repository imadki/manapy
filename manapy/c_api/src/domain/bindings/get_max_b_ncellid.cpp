// Bindings for get_max_b_ncellid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; b_visited is
// written in place (must start zeroed). Returns the distinct-cell count.
index_t get_max_b_ncellid_py(CIVec b_nodeid, CIMat node_cellid,
                             I8Vec b_visited) {
  return get_max_b_ncellid(make_view<const index_t, 1>(b_nodeid),
                            make_view<const index_t, 2>(node_cellid),
                            make_view<std::int8_t, 1>(b_visited));
}

} // namespace

void register_get_max_b_ncellid(nb::module_ &m) {
  m.def("get_max_b_ncellid", &get_max_b_ncellid_py, nb::arg("b_nodeid"),
        nb::arg("node_cellid"), nb::arg("b_visited").noconvert(),
        "Number of distinct cells touching any node in b_nodeid; used to "
        "size b_ncellid before create_b_ncellid. b_visited is scratch "
        "(must start zeroed) and is written in place.");
}
