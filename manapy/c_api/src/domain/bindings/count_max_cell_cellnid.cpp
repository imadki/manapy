// Bindings for count_max_cell_cellnid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; i_visited is
// written in place. Returns the max neighbor count.
index_t count_max_cell_cellnid_py(CIMat cells, CIMat node_cellid,
                                  IVec i_visited) {
  return count_max_cell_cellnid(make_view<const index_t, 2>(cells),
                                 make_view<const index_t, 2>(node_cellid),
                                 make_view<index_t, 1>(i_visited));
}

} // namespace

void register_count_max_cell_cellnid(nb::module_ &m) {
  m.def("count_max_cell_cellnid", &count_max_cell_cellnid_py, nb::arg("cells"),
        nb::arg("node_cellid"), nb::arg("i_visited").noconvert(),
        "For each cell, the number of distinct node-neighboring cells "
        "(excluding itself); returns the maximum across all cells, used to "
        "size cell_cellnid's row width before create_cell_cellnid. "
        "i_visited is scratch, sized to the number of cells, and is "
        "written in place.");
}
