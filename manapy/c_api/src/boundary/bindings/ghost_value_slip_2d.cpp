// Bindings for the 2D free-slip (slip wall) boundary condition on local
// boundary faces. This translation unit is compiled four times, once per
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
// which the C++ loop supplies); u_ghost/v_ghost are written in place.
void ghost_value_slip_2d_py(CFVec u_c, CFVec v_c, FVec u_ghost, FVec v_ghost,
                            CIMat face_cellid, CIVec bc_faces, CFMat normal) {
  ghost_value_slip_2d(
      make_view<const real_t, 1>(u_c), make_view<const real_t, 1>(v_c),
      make_view<real_t, 1>(u_ghost), make_view<real_t, 1>(v_ghost),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(bc_faces), make_view<const real_t, 2>(normal));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per bc_faces entry, and u_ghost/v_ghost are written in place on the GPU.
void ghost_value_slip_2d_cuda_py(DCFVec u_c, DCFVec v_c, DFVec u_ghost,
                                 DFVec v_ghost, DCIMat face_cellid,
                                 DCIVec bc_faces, DCFMat normal) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u_c.device_id()), "cudaSetDevice");

  launch_ghost_value_slip_2d(
      make_view<const real_t, 1>(u_c), make_view<const real_t, 1>(v_c),
      make_view<real_t, 1>(u_ghost), make_view<real_t, 1>(v_ghost),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(bc_faces), make_view<const real_t, 2>(normal),
      /*stream=*/nullptr);

  // Surface launch errors, then block until the in-place writes are visible.
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "ghost_value_slip_2d kernel");
}

} // namespace

void register_ghost_value_slip_2d(nb::module_ &m) {
  m.def("ghost_value_slip_2d", &ghost_value_slip_2d_py, nb::arg("u_c"),
        nb::arg("v_c"), nb::arg("u_ghost").noconvert(),
        nb::arg("v_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("normal"),
        "2D free-slip (slip wall) boundary condition on the faces listed in "
        "bc_faces: the velocity is reflected across the face, U_ghost = U_c - "
        "2 (U_c . n) n, so its normal component vanishes at the wall. The "
        "normal is normalised internally. Writes u_ghost/v_ghost in place.");

  m.def("ghost_value_slip_2d_cuda", &ghost_value_slip_2d_cuda_py,
        nb::arg("u_c"), nb::arg("v_c"), nb::arg("u_ghost").noconvert(),
        nb::arg("v_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("normal"),
        "GPU (CuPy) 2D free-slip boundary condition. Takes device arrays and "
        "writes u_ghost/v_ghost in place on the GPU.");
}
