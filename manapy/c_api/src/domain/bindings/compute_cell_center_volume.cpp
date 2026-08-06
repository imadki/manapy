// Bindings for compute_cell_center_area_2d / compute_cell_center_volume_3d.
// Compiled four times, once per manapy_compute_<float bits>_<int bits>
// package, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting
// the precisions.
//
// The halo_* counterparts declared alongside these in domain_compute.hpp are
// deliberately not bound: they exist for src/partitioning to call while
// building each subdomain's halo tables, and were never part of the c_api's
// Python surface either.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the c_api original; cell_area and
// cell_center are written in place.
void compute_cell_center_area_2d_py(CIMat cells, CFMat nodes, FVec cell_area,
                                    FMat cell_center) {
  compute_cell_center_area_2d(
      make_view<const index_t, 2>(cells), make_view<const real_t, 2>(nodes),
      make_view<real_t, 1>(cell_area), make_view<real_t, 2>(cell_center));
}

void compute_cell_center_volume_3d_py(CIMat cells, CFMat nodes,
                                      FVec cell_volume, FMat cell_center) {
  compute_cell_center_volume_3d(
      make_view<const index_t, 2>(cells), make_view<const real_t, 2>(nodes),
      make_view<real_t, 1>(cell_volume), make_view<real_t, 2>(cell_center));
}

} // namespace

void register_compute_cell_center_volume(nb::module_ &m) {
  // .noconvert() on the two OUTPUT arguments is load-bearing, not tidiness.
  // Without it, nanobind quietly casts a wrong-dtype array to a temporary,
  // the kernel fills the temporary, and the caller's array comes back
  // untouched -- wrong results, no error. (Passing a float32 cell_area to a
  // float64 build is exactly the mistake the transposed
  // manapy_part<INT>_<FLOAT> vs manapy_compute_<FLOAT>_<INT> naming invites.)
  // The read-only inputs keep implicit conversion, which is what the c_api's
  // PyArray_FROM_OTF did for them.
  m.def("compute_cell_center_area_2d", &compute_cell_center_area_2d_py,
        nb::arg("cells"), nb::arg("nodes"),
        nb::arg("cell_area").noconvert(), nb::arg("cell_center").noconvert(),
        "Centroid and area of every 2D cell (triangle or quad). cells' last "
        "column holds each row's node count. Writes cell_area (n_cells) and "
        "cell_center (n_cells, >=2) in place; returns None.");

  m.def("compute_cell_center_volume_3d", &compute_cell_center_volume_3d_py,
        nb::arg("cells"), nb::arg("nodes"),
        nb::arg("cell_volume").noconvert(), nb::arg("cell_center").noconvert(),
        "Centroid and volume of every 3D cell (tetrahedron, pyramid or "
        "hexahedron, decomposed into tetrahedra). cells' last column holds "
        "each row's node count. Writes cell_volume (n_cells) and cell_center "
        "(n_cells, >=3) in place; returns None.");
}
