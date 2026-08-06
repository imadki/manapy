// Bindings for create_bcell_halophyid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; i_visited and
// bcell_halophyid are written in place.
void create_bcell_halophyid_py(CIMat cells, CIVec b_ncellid,
                               CIMat node_halophyid, IVec i_visited,
                               IMat bcell_halophyid) {
  create_bcell_halophyid(make_view<const index_t, 2>(cells),
                          make_view<const index_t, 1>(b_ncellid),
                          make_view<const index_t, 2>(node_halophyid),
                          make_view<index_t, 1>(i_visited),
                          make_view<index_t, 2>(bcell_halophyid));
}

} // namespace

void register_create_bcell_halophyid(nb::module_ &m) {
  m.def("create_bcell_halophyid", &create_bcell_halophyid_py, nb::arg("cells"),
        nb::arg("b_ncellid"), nb::arg("node_halophyid"),
        nb::arg("i_visited").noconvert(),
        nb::arg("bcell_halophyid").noconvert(),
        "bcell_halophyid(i) = [cell global id, halo-phy-id, ..., count]. "
        "i_visited is scratch, same sizing as count_max_bcell_halophyid's "
        "(pass a freshly-zeroed buffer). Writes into i_visited and "
        "bcell_halophyid in place.");
}
