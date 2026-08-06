#include "advecdiff_compute.hpp"

#include "common/advecdiff/time_step_common.hpp"

// CPU entry point: minimum over all cells of the per-cell CFL candidate,
// seeded (like the Python original) at TIME_STEP_NO_LIMIT. face_measure and
// dim from the Python signature are unused by the computation and omitted here;
// Dxx/Dyy/Dzz add the diffusion term to lambda.
real_t time_step(ArrayView<const real_t, 1> u, ArrayView<const real_t, 1> v,
                 ArrayView<const real_t, 1> w, real_t cfl,
                 ArrayView<const real_t, 2> face_normal,
                 ArrayView<const real_t, 1> cell_volume,
                 ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                 real_t Dyy, real_t Dzz) {
  const index_t nbelement = static_cast<index_t>(cell_faceid.size(0));

  real_t dt = TIME_STEP_NO_LIMIT;
  for (index_t i = 0; i < nbelement; ++i) {
    const real_t cand = time_step_cell(i, u, v, w, cfl, face_normal,
                                       cell_volume, cell_faceid, Dxx, Dyy, Dzz);
    if (cand < dt)
      dt = cand;
  }
  return dt;
}
