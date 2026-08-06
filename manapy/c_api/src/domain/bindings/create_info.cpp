// Bindings for create_info. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; tmp_cell_faces,
// tmp_size_info, tmp_cell_faces_map, faces, cell_faceid, face_cellid,
// cell_cellfid and faces_counter are written in place.
void create_info_py(CIMat cells, CIMat node_cellid, CI8Vec cell_type,
                    IMat tmp_cell_faces, IVec tmp_size_info,
                    IMat tmp_cell_faces_map, IMat faces, IMat cell_faceid,
                    IMat face_cellid, IMat cell_cellfid, IVec faces_counter) {
  create_info(make_view<const index_t, 2>(cells),
              make_view<const index_t, 2>(node_cellid),
              make_view<const std::int8_t, 1>(cell_type),
              make_view<index_t, 2>(tmp_cell_faces),
              make_view<index_t, 1>(tmp_size_info),
              make_view<index_t, 2>(tmp_cell_faces_map),
              make_view<index_t, 2>(faces),
              make_view<index_t, 2>(cell_faceid),
              make_view<index_t, 2>(face_cellid),
              make_view<index_t, 2>(cell_cellfid),
              make_view<index_t, 1>(faces_counter));
}

} // namespace

void register_create_info(nb::module_ &m) {
  m.def(
      "create_info", &create_info_py, nb::arg("cells"), nb::arg("node_cellid"),
      nb::arg("cell_type"), nb::arg("tmp_cell_faces").noconvert(),
      nb::arg("tmp_size_info").noconvert(),
      nb::arg("tmp_cell_faces_map").noconvert(), nb::arg("faces").noconvert(),
      nb::arg("cell_faceid").noconvert(), nb::arg("face_cellid").noconvert(),
      nb::arg("cell_cellfid").noconvert(), nb::arg("faces_counter").noconvert(),
      "Builds faces, cell->face, face->cell and cell->neighbor-cell (by "
      "shared face) tables in one pass over cells. tmp_cell_faces and "
      "tmp_size_info are scratch, overwritten per cell; "
      "tmp_cell_faces_map persists across the call (one row per cell) to "
      "let a shared face be found instead of duplicated. faces, "
      "cell_faceid, face_cellid, cell_cellfid and faces_counter are "
      "written in place; cell_faceid, cell_cellfid and faces_counter "
      "must start zeroed.");
}
