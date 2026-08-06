// Bindings for create_cellfid. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original, minus
// max_cell_faceid/max_face_nodeid/tmp_cell_faces/tmp_size_info: those sized
// and passed a per-cell scratch buffer for numba.prange in the original;
// this OpenMP-parallelized port allocates that scratch itself, per cell,
// inside the loop (see cpu/create_cellfid_cpu.cpp), so there's nothing for
// the caller to provide. cell_cellfid is written in place.
void create_cellfid_py(CIMat cells, CIMat node_cellid, CI8Vec cell_type,
                       IMat cell_cellfid) {
  create_cellfid(make_view<const index_t, 2>(cells),
                  make_view<const index_t, 2>(node_cellid),
                  make_view<const std::int8_t, 1>(cell_type),
                  make_view<index_t, 2>(cell_cellfid));
}

} // namespace

void register_create_cellfid(nb::module_ &m) {
  m.def("create_cellfid", &create_cellfid_py, nb::arg("cells"),
        nb::arg("node_cellid"), nb::arg("cell_type"),
        nb::arg("cell_cellfid").noconvert(),
        "Cell->neighbor-cell (by shared face) table, computed directly per "
        "cell without building create_info's faces/cell_faceid/face_cellid "
        "tables. Parallelized over cells with OpenMP. Writes into "
        "cell_cellfid in place; it must already be zeroed.");
}
