// Bindings for create_ghost_tables. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; ghost_i_visited,
// node_ghostid and cell_ghostid are written in place.
void create_ghost_tables_py(CIMat ghost_info_int, CIMat faces,
                            CIMat cell_faceid, CIMat node_cellid,
                            IVec ghost_i_visited, IMat node_ghostid,
                            IMat cell_ghostid) {
  create_ghost_tables(make_view<const index_t, 2>(ghost_info_int),
                       make_view<const index_t, 2>(faces),
                       make_view<const index_t, 2>(cell_faceid),
                       make_view<const index_t, 2>(node_cellid),
                       make_view<index_t, 1>(ghost_i_visited),
                       make_view<index_t, 2>(node_ghostid),
                       make_view<index_t, 2>(cell_ghostid));
}

} // namespace

void register_create_ghost_tables(nb::module_ &m) {
  m.def("create_ghost_tables", &create_ghost_tables_py,
        nb::arg("ghost_info_int"), nb::arg("faces"), nb::arg("cell_faceid"),
        nb::arg("node_cellid"), nb::arg("ghost_i_visited").noconvert(),
        nb::arg("node_ghostid").noconvert(),
        nb::arg("cell_ghostid").noconvert(),
        "node_ghostid/cell_ghostid: indices into ghost_info_int for the "
        "ghost cells neighboring each node/cell. ghost_i_visited is "
        "scratch, sized to the number of cells. Writes into "
        "ghost_i_visited, node_ghostid and cell_ghostid in place.");
}
