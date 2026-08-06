// _core module entry point. Defines NB_MODULE once and delegates to each
// kernel's register_* function (bindings/registry.hpp), keeping the kernels in
// separate translation units. Compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy compute kernels compiled for float" MANAPY_COMPUTE_STR(
      MANAPY_COMPUTE_FLOAT_BITS) " data and int" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS)
      " indices";

  register_cell_gradient_2d(m);
  register_center_to_vertex_2d(m);
  register_face_gradient_2d(m);
  register_barthlimiter_2d(m);
  register_vanalbadalimiter_2d(m);
  register_face_gradient_3d(m);
  register_cell_gradient_3d(m);
  register_center_to_vertex_3d(m);
  register_barthlimiter_3d(m);
  register_vanalbadalimiter_3d(m);
  register_facetocell(m);
  register_celltoface(m);
}
