// Bindings for the forward-Euler cell-field update, a utility common to all
// solvers (shipped as solvers.utils.update_new_value). Compiled four times, once
// per manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "bindings/registry.hpp"
#include "utils_compute.cuh"
#include "utils_compute.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original; ne_c is updated in
// place.
void update_new_value_py(FVec ne_c, CFVec rez_ne, CFVec dissip_ne,
                         CFVec src_ne, real_t dtime, CFVec cell_volume) {
  update_new_value(make_view<real_t, 1>(ne_c),
                   make_view<const real_t, 1>(rez_ne),
                   make_view<const real_t, 1>(dissip_ne),
                   make_view<const real_t, 1>(src_ne), dtime,
                   make_view<const real_t, 1>(cell_volume));
}

// GPU version: same signature/argument order, but every array is a CuPy device
// array ingested zero-copy via DLPack. ne_c is updated in place on the GPU.
void update_new_value_cuda_py(DFVec ne_c, DCFVec rez_ne, DCFVec dissip_ne,
                              DCFVec src_ne, real_t dtime, DCFVec cell_volume) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(ne_c.device_id()), "cudaSetDevice");

  launch_update_new_value(make_view<real_t, 1>(ne_c),
                          make_view<const real_t, 1>(rez_ne),
                          make_view<const real_t, 1>(dissip_ne),
                          make_view<const real_t, 1>(src_ne), dtime,
                          make_view<const real_t, 1>(cell_volume),
                          /*stream=*/nullptr);

  // Surface launch errors, then block until the in-place writes are visible.
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "update_new_value kernel");
}

} // namespace

void register_update_new_value(nb::module_ &m) {
  m.def("update_new_value", &update_new_value_py, nb::arg("ne_c").noconvert(),
        nb::arg("rez_ne"), nb::arg("dissip_ne"), nb::arg("src_ne"),
        nb::arg("dtime"), nb::arg("cell_volume"),
        "Explicit forward-Euler update of a cell field: ne_c += dtime * "
        "((rez + dissip) / volume + src). Updates ne_c in place.");

  m.def("update_new_value_cuda", &update_new_value_cuda_py,
        nb::arg("ne_c").noconvert(), nb::arg("rez_ne"), nb::arg("dissip_ne"),
        nb::arg("src_ne"), nb::arg("dtime"), nb::arg("cell_volume"),
        "GPU (CuPy) explicit forward-Euler update of a cell field. Takes "
        "device arrays and updates ne_c in place on the GPU.");
}
