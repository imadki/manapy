// Bindings for create_b_ncellid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; b_visited and
// b_ncellid are written in place (b_visited must start zeroed and be a
// fresh buffer, not the one already consumed by get_max_b_ncellid).
void create_b_ncellid_py(CIVec b_nodeid, CIMat node_cellid, I8Vec b_visited,
                         IVec b_ncellid) {
  create_b_ncellid(make_view<const index_t, 1>(b_nodeid),
                    make_view<const index_t, 2>(node_cellid),
                    make_view<std::int8_t, 1>(b_visited),
                    make_view<index_t, 1>(b_ncellid));
}

} // namespace

void register_create_b_ncellid(nb::module_ &m) {
  m.def("create_b_ncellid", &create_b_ncellid_py, nb::arg("b_nodeid"),
        nb::arg("node_cellid"), nb::arg("b_visited").noconvert(),
        nb::arg("b_ncellid").noconvert(),
        "The distinct cells touching any node in b_nodeid, written into "
        "b_ncellid. b_visited is scratch (must start zeroed, and must be a "
        "fresh buffer rather than the one get_max_b_ncellid already "
        "consumed).");
}
