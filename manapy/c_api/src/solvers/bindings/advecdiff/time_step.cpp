// Bindings for the explicit CFL time-step reduction of the advection-diffusion
// solver (advec's convective reduction plus a diffusion term). Compiled four
// times, once per manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "advecdiff_compute.cuh"
#include "advecdiff_compute.hpp"
#include "bindings/registry.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original; returns the time step.
// face_measure and dim are unused by the computation but kept for parity.
real_t time_step_py(CFVec u, CFVec v, CFVec w, real_t cfl, CFMat face_normal,
                    CFVec face_measure, CFVec cell_volume, CIMat cell_faceid,
                    index_t dim, real_t Dxx, real_t Dyy, real_t Dzz) {
  (void)face_measure; // unused by the computation, kept for signature parity
  (void)dim;          // unused by the computation, kept for signature parity

  return time_step(make_view<const real_t, 1>(u), make_view<const real_t, 1>(v),
                   make_view<const real_t, 1>(w), cfl,
                   make_view<const real_t, 2>(face_normal),
                   make_view<const real_t, 1>(cell_volume),
                   make_view<const index_t, 2>(cell_faceid), Dxx, Dyy, Dzz);
}

// GPU version: same signature/argument order, but every array is a CuPy device
// array ingested zero-copy via DLPack. The reduction runs on the GPU and the
// scalar result is copied back and returned.
real_t time_step_cuda_py(DCFVec u, DCFVec v, DCFVec w, real_t cfl,
                         DCFMat face_normal, DCFVec face_measure,
                         DCFVec cell_volume, DCIMat cell_faceid, index_t dim,
                         real_t Dxx, real_t Dyy, real_t Dzz) {
  (void)face_measure; // unused by the computation, kept for signature parity
  (void)dim;          // unused by the computation, kept for signature parity

  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u.device_id()), "cudaSetDevice");

  const real_t dt = launch_time_step(
      make_view<const real_t, 1>(u), make_view<const real_t, 1>(v),
      make_view<const real_t, 1>(w), cfl,
      make_view<const real_t, 2>(face_normal),
      make_view<const real_t, 1>(cell_volume),
      make_view<const index_t, 2>(cell_faceid), Dxx, Dyy, Dzz,
      /*stream=*/nullptr);

  // launch_time_step already synchronised; surface any error it left pending.
  cuda_check(cudaGetLastError(), "time_step kernel");
  return dt;
}

} // namespace

void register_advecdiff_time_step(nb::module_ &m) {
  m.def("time_step", &time_step_py, nb::arg("u"), nb::arg("v"), nb::arg("w"),
        nb::arg("cfl"), nb::arg("face_normal"), nb::arg("face_measure"),
        nb::arg("cell_volume"), nb::arg("cell_faceid"), nb::arg("dim"),
        nb::arg("Dxx"), nb::arg("Dyy"), nb::arg("Dzz"),
        "Explicit CFL time step for the advection-diffusion solver: min over "
        "all cells of cfl * cell_volume / lambda, where lambda sums |u.n| and "
        "the diffusion term (Dxx+Dyy+Dzz)*||n||^2/volume over the cell's faces. "
        "Returns the time step.");

  m.def("time_step_cuda", &time_step_cuda_py, nb::arg("u"), nb::arg("v"),
        nb::arg("w"), nb::arg("cfl"), nb::arg("face_normal"),
        nb::arg("face_measure"), nb::arg("cell_volume"), nb::arg("cell_faceid"),
        nb::arg("dim"), nb::arg("Dxx"), nb::arg("Dyy"), nb::arg("Dzz"),
        "GPU (CuPy) explicit CFL time step for the advection-diffusion solver. "
        "Takes device arrays, reduces on the GPU and returns the time step.");
}
