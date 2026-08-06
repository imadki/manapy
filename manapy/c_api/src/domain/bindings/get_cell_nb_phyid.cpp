// Bindings for get_cell_nb_phyid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; i_visited and
// cell_nb_phyid are written in place.
void get_cell_nb_phyid_py(CIMat phy_faces, CIMat node_cellid, IVec i_visited,
                          IVec cell_nb_phyid) {
  get_cell_nb_phyid(make_view<const index_t, 2>(phy_faces),
                     make_view<const index_t, 2>(node_cellid),
                     make_view<index_t, 1>(i_visited),
                     make_view<index_t, 1>(cell_nb_phyid));
}

} // namespace

void register_get_cell_nb_phyid(nb::module_ &m) {
  m.def("get_cell_nb_phyid", &get_cell_nb_phyid_py, nb::arg("phy_faces"),
        nb::arg("node_cellid"), nb::arg("i_visited").noconvert(),
        nb::arg("cell_nb_phyid").noconvert(),
        "Increments cell_nb_phyid[cell] once per physical face that has one "
        "of cell's nodes on it. i_visited is scratch, sized to the number "
        "of cells. Writes into i_visited and cell_nb_phyid in place.");
}
