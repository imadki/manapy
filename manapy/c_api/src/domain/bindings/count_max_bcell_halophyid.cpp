// Bindings for count_max_bcell_halophyid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; i_visited is
// written in place. Returns the max halo-phyid count.
index_t count_max_bcell_halophyid_py(CIMat cells, CIVec b_ncellid,
                                     CIMat node_halophyid, IVec i_visited) {
  return count_max_bcell_halophyid(make_view<const index_t, 2>(cells),
                                    make_view<const index_t, 1>(b_ncellid),
                                    make_view<const index_t, 2>(node_halophyid),
                                    make_view<index_t, 1>(i_visited));
}

} // namespace

void register_count_max_bcell_halophyid(nb::module_ &m) {
  m.def("count_max_bcell_halophyid", &count_max_bcell_halophyid_py,
        nb::arg("cells"), nb::arg("b_ncellid"), nb::arg("node_halophyid"),
        nb::arg("i_visited").noconvert(),
        "For each boundary cell (indexed via b_ncellid), the number of "
        "distinct halo-physical-face ids touching its nodes; returns the "
        "maximum, used to size bcell_halophyid's row width before "
        "create_bcell_halophyid. i_visited is scratch and is written in "
        "place.");
}
