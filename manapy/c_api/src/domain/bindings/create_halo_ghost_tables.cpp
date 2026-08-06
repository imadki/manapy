// Bindings for create_halo_ghost_tables. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; ext_ghost_info_int,
// cell_haloghostid and node_haloghostid are written in place.
void create_halo_ghost_tables_py(IMat ext_ghost_info_int,
                                 CIVec node_halophyid, CIVec cell_halophyid,
                                 CIMat node_haloid, CIMat halo_halosext,
                                 IMat cell_haloghostid,
                                 IMat node_haloghostid) {
  create_halo_ghost_tables(
      make_view<index_t, 2>(ext_ghost_info_int),
      make_view<const index_t, 1>(node_halophyid),
      make_view<const index_t, 1>(cell_halophyid),
      make_view<const index_t, 2>(node_haloid),
      make_view<const index_t, 2>(halo_halosext),
      make_view<index_t, 2>(cell_haloghostid),
      make_view<index_t, 2>(node_haloghostid));
}

} // namespace

void register_create_halo_ghost_tables(nb::module_ &m) {
  m.def("create_halo_ghost_tables", &create_halo_ghost_tables_py,
        nb::arg("ext_ghost_info_int").noconvert(), nb::arg("node_halophyid"),
        nb::arg("cell_halophyid"), nb::arg("node_haloid"),
        nb::arg("halo_halosext"), nb::arg("cell_haloghostid").noconvert(),
        nb::arg("node_haloghostid").noconvert(),
        "Unpacks the flat-encoded cell_halophyid/node_halophyid (format: "
        "[id1, size1, val1_1, ..., id2, size2, ...]) into "
        "cell_haloghostid/node_haloghostid, each row's entries being "
        "indices into ext_ghost_info_int. Also patches "
        "ext_ghost_info_int(:, 0) in place, resolving it to the local "
        "halo-cell index found via a global-id lookup. Writes into "
        "ext_ghost_info_int, cell_haloghostid and node_haloghostid in "
        "place.");
}
