// Bindings for node_periodic_bits. Compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_bits is
// written in place (must already be zeroed).
void node_periodic_bits_py(CIMat faces, CIVec face_name, IVec node_bits) {
  node_periodic_bits(make_view<const index_t, 2>(faces),
                     make_view<const index_t, 1>(face_name),
                     make_view<index_t, 1>(node_bits));
}

} // namespace

void register_node_periodic_bits(nb::module_ &m) {
  m.def("node_periodic_bits", &node_periodic_bits_py, nb::arg("faces"),
        nb::arg("face_name"), nb::arg("node_bits").noconvert(),
        "Per-node bitmask of the periodic boundaries the node lies on, "
        "taken from the periodic faces it is a vertex of "
        "(1=11 2=22 4=33 8=44 16=55 32=66). Writes into node_bits in place; "
        "node_bits must already be zeroed.");
}
