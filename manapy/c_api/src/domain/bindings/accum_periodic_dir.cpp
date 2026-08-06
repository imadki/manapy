// Bindings for accum_periodic_dir. Compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_periodicid and
// node_fill are written in place.
void accum_periodic_dir_py(CIVec node_bits, CFMat nodes, CIMat node_cellid,
                           IMat node_periodicid, IVec node_fill, CFVec cmin,
                           index_t lo_bit, index_t hi_bit, index_t taxis0,
                           index_t taxis1, real_t dtol) {
  accum_periodic_dir(make_view<const index_t, 1>(node_bits),
                     make_view<const real_t, 2>(nodes),
                     make_view<const index_t, 2>(node_cellid),
                     make_view<index_t, 2>(node_periodicid),
                     make_view<index_t, 1>(node_fill),
                     make_view<const real_t, 1>(cmin), lo_bit, hi_bit, taxis0,
                     taxis1, dtol);
}

} // namespace

void register_accum_periodic_dir(nb::module_ &m) {
  m.def("accum_periodic_dir", &accum_periodic_dir_py, nb::arg("node_bits"),
        nb::arg("nodes"), nb::arg("node_cellid"),
        nb::arg("node_periodicid").noconvert(),
        nb::arg("node_fill").noconvert(), nb::arg("cmin"), nb::arg("lo_bit"),
        nb::arg("hi_bit"), nb::arg("taxis0"), nb::arg("taxis1"),
        nb::arg("dtol"),
        "For ONE periodic axis (from node_periodic_bits' per-axis "
        "lo_bit/hi_bit pair), matches boundary nodes carrying lo_bit to "
        "those carrying hi_bit by their transverse coordinate(s) "
        "taxis0[,taxis1] (pass taxis1=-1 for a single transverse axis), "
        "and appends each side's partner cells into node_periodicid "
        "in place (per-node running counter node_fill, which must persist "
        "and accumulate across repeated calls for different axes). "
        "Unmatched (cross-rank) nodes are silently left unpaired.");
}
