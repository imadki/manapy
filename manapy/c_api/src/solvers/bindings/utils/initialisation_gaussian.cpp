// Bindings for the Gaussian initial-condition kernels (2D and 3D). CPU-only:
// no device (_cuda) variant. Compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "utils_compute.hpp"

namespace {

// Same signature/argument order as the Python original; ne, u, v, P are written
// in place.
void initialisation_gaussian_2d_py(FVec ne, FVec u, FVec v, FVec P,
                                   CFMat cell_center, real_t Pinit) {
  initialisation_gaussian_2d(
      make_view<real_t, 1>(ne), make_view<real_t, 1>(u),
      make_view<real_t, 1>(v), make_view<real_t, 1>(P),
      make_view<const real_t, 2>(cell_center), Pinit);
}

// Same signature/argument order as the Python original; ne, u, v, w, P are
// written in place.
void initialisation_gaussian_3d_py(FVec ne, FVec u, FVec v, FVec w, FVec P,
                                   CFMat cell_center, real_t Pinit) {
  initialisation_gaussian_3d(
      make_view<real_t, 1>(ne), make_view<real_t, 1>(u),
      make_view<real_t, 1>(v), make_view<real_t, 1>(w),
      make_view<real_t, 1>(P), make_view<const real_t, 2>(cell_center), Pinit);
}

} // namespace

void register_initialisation_gaussian(nb::module_ &m) {
  m.def("initialisation_gaussian_2d", &initialisation_gaussian_2d_py,
        nb::arg("ne").noconvert(), nb::arg("u").noconvert(),
        nb::arg("v").noconvert(), nb::arg("P").noconvert(),
        nb::arg("cell_center"), nb::arg("Pinit"),
        "Gaussian bump initial condition on a 2D mesh: ne = Gaussian centred "
        "at (0.2, 0.2), u = v = 0, P = Pinit * (0.5 - x). Written in place.");

  m.def("initialisation_gaussian_3d", &initialisation_gaussian_3d_py,
        nb::arg("ne").noconvert(), nb::arg("u").noconvert(),
        nb::arg("v").noconvert(), nb::arg("w").noconvert(),
        nb::arg("P").noconvert(), nb::arg("cell_center"), nb::arg("Pinit"),
        "Gaussian bump initial condition on a 3D mesh: ne = Gaussian centred "
        "at (0.2, 0.25, 0.45), u = v = w = 0, P = Pinit * (0.5 - x). Written "
        "in place.");
}
