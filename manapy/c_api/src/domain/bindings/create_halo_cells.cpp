// Bindings for create_halo_cells. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; node_haloid,
// b_visited, cell_halonid and face_haloid are written in place.
void create_halo_cells_py(CIMat cells, CIMat faces, CIVec node_halos,
                          IMat node_haloid, I8Vec b_visited,
                          IMat cell_halonid, IVec face_haloid) {
  create_halo_cells(make_view<const index_t, 2>(cells),
                     make_view<const index_t, 2>(faces),
                     make_view<const index_t, 1>(node_halos),
                     make_view<index_t, 2>(node_haloid),
                     make_view<std::int8_t, 1>(b_visited),
                     make_view<index_t, 2>(cell_halonid),
                     make_view<index_t, 1>(face_haloid));
}

} // namespace

void register_create_halo_cells(nb::module_ &m) {
  m.def("create_halo_cells", &create_halo_cells_py, nb::arg("cells"),
        nb::arg("faces"), nb::arg("node_halos"),
        nb::arg("node_haloid").noconvert(), nb::arg("b_visited").noconvert(),
        nb::arg("cell_halonid").noconvert(), nb::arg("face_haloid").noconvert(),
        "Unpacks the flat-encoded node_halos into node_haloid (must start "
        "zeroed), resolves each face's single bordering halo cell (or -1) "
        "into face_haloid, and collects each cell's deduplicated halo "
        "neighbors into cell_halonid (must start zeroed). b_visited is "
        "scratch. Writes into node_haloid, b_visited, cell_halonid and "
        "face_haloid in place.");
}
