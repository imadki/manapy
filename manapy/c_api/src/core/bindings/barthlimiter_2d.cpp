// Bindings for the Barth-Jespersen limiter kernel. This translation unit is
// compiled four times, once per manapy_compute_<float bits>_<int bits>
// package, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting
// the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "bindings/registry.hpp"
#include "variable_compute.cuh"
#include "variable_compute.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original; psi is written in
// place.
void barthlimiter_2d_py(CFVec w_c, CFVec w_ghost, CFVec w_halo, CFVec w_x,
                        CFVec w_y, CFVec w_z, FVec psi, CIMat face_cellid,
                        CIMat cell_faceid, CIVec face_name, CIVec face_haloid,
                        CFMat cell_center, CFMat face_center) {
  (void)w_z; // unused by the kernel, kept for signature parity

  barthlimiter_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo), make_view<const real_t, 1>(w_x),
      make_view<const real_t, 1>(w_y), make_view<real_t, 1>(psi),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 2>(cell_faceid),
      make_view<const index_t, 1>(face_name),
      make_view<const index_t, 1>(face_haloid),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(face_center));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one
// thread per cell, and psi is written in place on the GPU.
void barthlimiter_2d_cuda_py(DCFVec w_c, DCFVec w_ghost, DCFVec w_halo,
                             DCFVec w_x, DCFVec w_y, DCFVec w_z, DFVec psi,
                             DCIMat face_cellid, DCIMat cell_faceid,
                             DCIVec face_name, DCIVec face_haloid,
                             DCFMat cell_center, DCFMat face_center) {
  (void)w_z; // unused by the kernel, kept for signature parity

  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(w_c.device_id()), "cudaSetDevice");

  launch_barthlimiter_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo), make_view<const real_t, 1>(w_x),
      make_view<const real_t, 1>(w_y), make_view<real_t, 1>(psi),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 2>(cell_faceid),
      make_view<const index_t, 1>(face_name),
      make_view<const index_t, 1>(face_haloid),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(face_center), /*stream=*/nullptr);

  // Surface launch errors, then block until the in-place writes are visible.
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "barthlimiter_2d kernel");
}

} // namespace

void register_barthlimiter_2d(nb::module_ &m) {
  m.def("barthlimiter_2d", &barthlimiter_2d_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_x"), nb::arg("w_y"),
        nb::arg("w_z"), nb::arg("psi").noconvert(), nb::arg("face_cellid"),
        nb::arg("cell_faceid"), nb::arg("face_name"), nb::arg("face_haloid"),
        nb::arg("cell_center"), nb::arg("face_center"),
        "Barth-Jespersen slope limiter for a reconstructed cell gradient on a "
        "2D unstructured mesh. Writes the result into psi in place.");

  m.def("barthlimiter_2d_cuda", &barthlimiter_2d_cuda_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_x"), nb::arg("w_y"),
        nb::arg("w_z"), nb::arg("psi").noconvert(), nb::arg("face_cellid"),
        nb::arg("cell_faceid"), nb::arg("face_name"), nb::arg("face_haloid"),
        nb::arg("cell_center"), nb::arg("face_center"),
        "GPU (CuPy) Barth-Jespersen slope limiter for a reconstructed cell "
        "gradient on a 2D unstructured mesh. Takes device arrays and writes "
        "the result into psi in place on the GPU.");
}
