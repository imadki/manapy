// Bindings for the 2D free-slip (slip wall) boundary condition on halo ghosts.
// This translation unit is compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "bindings/registry.hpp"
#include "boundary_compute.cuh"
#include "boundary_compute.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original (minus start/stride,
// which the C++ loop supplies); u_haloghost/v_haloghost written in place.
void haloghost_value_slip_2d_py(CFVec u_halo, CFVec v_halo, FVec u_haloghost,
                                FVec v_haloghost, CIMat node_haloghostid,
                                CIMat ghost_ext_info_int,
                                CFMat ghost_ext_info_flt, index_t BCindex,
                                CIVec d_halonodes) {
  haloghost_value_slip_2d(make_view<const real_t, 1>(u_halo),
                          make_view<const real_t, 1>(v_halo),
                          make_view<real_t, 1>(u_haloghost),
                          make_view<real_t, 1>(v_haloghost),
                          make_view<const index_t, 2>(node_haloghostid),
                          make_view<const index_t, 2>(ghost_ext_info_int),
                          make_view<const real_t, 2>(ghost_ext_info_flt),
                          BCindex, make_view<const index_t, 1>(d_halonodes));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per d_halonodes entry, and the halo-ghost components are written in place on
// the GPU.
void haloghost_value_slip_2d_cuda_py(DCFVec u_halo, DCFVec v_halo,
                                     DFVec u_haloghost, DFVec v_haloghost,
                                     DCIMat node_haloghostid,
                                     DCIMat ghost_ext_info_int,
                                     DCFMat ghost_ext_info_flt, index_t BCindex,
                                     DCIVec d_halonodes) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u_halo.device_id()), "cudaSetDevice");

  launch_haloghost_value_slip_2d(
      make_view<const real_t, 1>(u_halo), make_view<const real_t, 1>(v_halo),
      make_view<real_t, 1>(u_haloghost), make_view<real_t, 1>(v_haloghost),
      make_view<const index_t, 2>(node_haloghostid),
      make_view<const index_t, 2>(ghost_ext_info_int),
      make_view<const real_t, 2>(ghost_ext_info_flt), BCindex,
      make_view<const index_t, 1>(d_halonodes), /*stream=*/nullptr);

  // Surface launch errors, then block until the in-place writes are visible.
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "haloghost_value_slip_2d kernel");
}

} // namespace

void register_haloghost_value_slip_2d(nb::module_ &m) {
  m.def("haloghost_value_slip_2d", &haloghost_value_slip_2d_py,
        nb::arg("u_halo"), nb::arg("v_halo"),
        nb::arg("u_haloghost").noconvert(), nb::arg("v_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"),
        "2D free-slip (slip wall) boundary condition on every halo ghost "
        "tagged BCindex that hangs off a node of d_halonodes: the halo "
        "velocity is reflected across the face, U = U - 2 (U . n) n, with the "
        "normal read from columns 7-8 of ghost_ext_info_flt and normalised "
        "internally. Writes u_haloghost/v_haloghost in place.");

  m.def("haloghost_value_slip_2d_cuda", &haloghost_value_slip_2d_cuda_py,
        nb::arg("u_halo"), nb::arg("v_halo"),
        nb::arg("u_haloghost").noconvert(), nb::arg("v_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"),
        "GPU (CuPy) 2D free-slip boundary condition on halo ghosts. Takes "
        "device arrays and writes u_haloghost/v_haloghost in place on the "
        "GPU.");
}
