// Bindings for define_node_oldname. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_oldname is
// written in place.
void define_node_oldname_py(CIMat phy_faces, CIVec phy_faces_name,
                            IVec node_oldname) {
  define_node_oldname(make_view<const index_t, 2>(phy_faces),
                       make_view<const index_t, 1>(phy_faces_name),
                       make_view<index_t, 1>(node_oldname));
}

} // namespace

void register_define_node_oldname(nb::module_ &m) {
  m.def("define_node_oldname", &define_node_oldname_py, nb::arg("phy_faces"),
        nb::arg("phy_faces_name"), nb::arg("node_oldname").noconvert(),
        "For each node touched by a physical face, the smallest "
        "physical-face name among all physical faces touching it. Writes "
        "into node_oldname in place.");
}
