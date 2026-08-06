// Bindings for the 3D free-slip (slip wall) boundary condition on local
// boundary faces. This translation unit is compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include "cuda_launch.hpp"

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
// which the C++ loop supplies); u_ghost/v_ghost/w_ghost are written in place.
void ghost_value_slip_3d_py(CFVec u_c, CFVec v_c, CFVec w_c, FVec u_ghost,
                            FVec v_ghost, FVec w_ghost, CIMat face_cellid,
                            CIVec bc_faces, CFMat normal) {
  ghost_value_slip_3d(
      make_view<const real_t, 1>(u_c), make_view<const real_t, 1>(v_c),
      make_view<const real_t, 1>(w_c), make_view<real_t, 1>(u_ghost),
      make_view<real_t, 1>(v_ghost), make_view<real_t, 1>(w_ghost),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(bc_faces), make_view<const real_t, 2>(normal));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per bc_faces entry, and the ghost components are written in place on the GPU.
void ghost_value_slip_3d_cuda_py(DCFVec u_c, DCFVec v_c, DCFVec w_c,
                                 DFVec u_ghost, DFVec v_ghost, DFVec w_ghost,
                                 DCIMat face_cellid, DCIVec bc_faces,
                                 DCFMat normal) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u_c.device_id()), "cudaSetDevice");

  launch_ghost_value_slip_3d(
      make_view<const real_t, 1>(u_c), make_view<const real_t, 1>(v_c),
      make_view<const real_t, 1>(w_c), make_view<real_t, 1>(u_ghost),
      make_view<real_t, 1>(v_ghost), make_view<real_t, 1>(w_ghost),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(bc_faces), make_view<const real_t, 2>(normal),
      /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), "ghost_value_slip_3d kernel launch");
}

} // namespace

void register_ghost_value_slip_3d(nb::module_ &m) {
  m.def("ghost_value_slip_3d", &ghost_value_slip_3d_py, nb::arg("u_c"),
        nb::arg("v_c"), nb::arg("w_c"), nb::arg("u_ghost").noconvert(),
        nb::arg("v_ghost").noconvert(), nb::arg("w_ghost").noconvert(),
        nb::arg("face_cellid"), nb::arg("bc_faces"), nb::arg("normal"),
        "3D free-slip (slip wall) boundary condition on the faces listed in "
        "bc_faces: the velocity is reflected across the face, U_ghost = U_c - "
        "2 (U_c . n) n, so its normal component vanishes at the wall. The "
        "normal is normalised internally. Writes u_ghost/v_ghost/w_ghost in "
        "place.");

  m.def("ghost_value_slip_3d_cuda", &ghost_value_slip_3d_cuda_py,
        nb::arg("u_c"), nb::arg("v_c"), nb::arg("w_c"),
        nb::arg("u_ghost").noconvert(), nb::arg("v_ghost").noconvert(),
        nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("normal"),
        "GPU (CuPy) 3D free-slip boundary condition. Takes device arrays and "
        "writes u_ghost/v_ghost/w_ghost in place on the GPU.");
}
