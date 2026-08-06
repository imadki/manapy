#include <cmath>

#include "utils_compute.hpp"

// CPU-only (see headers/utils/utils_compute.hpp): the whole kernel lives here.
// Sets a Gaussian bump in ne centred at (0.2, 0.25, 0.45), zeroes the velocity
// and sets a linear pressure P = Pinit * (0.5 - x). Written in place per cell.
void initialisation_gaussian_3d(ArrayView<real_t, 1> ne, ArrayView<real_t, 1> u,
                                ArrayView<real_t, 1> v, ArrayView<real_t, 1> w,
                                ArrayView<real_t, 1> P,
                                ArrayView<const real_t, 2> cell_center,
                                real_t Pinit) {
  const index_t nbelements = static_cast<index_t>(cell_center.size(0));

  const real_t sigma = real_t(0.05);
  const real_t sigma2 = sigma * sigma;

  for (index_t i = 0; i < nbelements; ++i) {
    const real_t xcent = cell_center(i, 0);
    const real_t ycent = cell_center(i, 1);
    const real_t zcent = cell_center(i, 2);

    const real_t dx = xcent - real_t(0.2);
    const real_t dy = ycent - real_t(0.25);
    const real_t dz = zcent - real_t(0.45);

    ne(i) = real_t(5) *
                std::exp(real_t(-1) * (dx * dx + dy * dy + dz * dz) / sigma2) +
            real_t(1);
    u(i) = real_t(0);
    v(i) = real_t(0);
    w(i) = real_t(0);
    P(i) = Pinit * (real_t(0.5) - xcent);
  }
}
