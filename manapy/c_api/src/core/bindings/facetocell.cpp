// Bindings for the face-to-cell averaging kernel. This translation unit is
// compiled four times, once per manapy_compute_<float bits>_<int bits>
// package, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting
// the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include "cuda_launch.hpp"

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

// Same signature/argument order as the Python original; u_c is written in
// place.
void facetocell_py(CFVec u_face, FVec u_c, CIMat cell_faceid, index_t dim) {
  (void)dim; // unused by the kernel, kept for signature parity

  facetocell(make_view<const real_t, 1>(u_face),
             make_view<const index_t, 2>(cell_faceid),
             make_view<real_t, 1>(u_c));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one
// thread per cell, and u_c is written in place on the GPU.
void facetocell_cuda_py(DCFVec u_face, DFVec u_c, DCIMat cell_faceid,
                        index_t dim) {
  (void)dim; // unused by the kernel, kept for signature parity

  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u_face.device_id()), "cudaSetDevice");

  launch_facetocell(make_view<const real_t, 1>(u_face),
                     make_view<const index_t, 2>(cell_faceid),
                     make_view<real_t, 1>(u_c), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), "facetocell kernel launch");
}

} // namespace

void register_facetocell(nb::module_ &m) {
  m.def("facetocell", &facetocell_py, nb::arg("u_face"),
        nb::arg("u_c").noconvert(), nb::arg("cell_faceid"), nb::arg("dim"),
        "Face-to-cell averaging of a face field onto cells on a 2D/3D "
        "unstructured mesh. Writes the result into u_c in place.");

  m.def("facetocell_cuda", &facetocell_cuda_py, nb::arg("u_face"),
        nb::arg("u_c").noconvert(), nb::arg("cell_faceid"), nb::arg("dim"),
        "GPU (CuPy) face-to-cell averaging of a face field onto cells on a "
        "2D/3D unstructured mesh. Takes device arrays and writes the result "
        "into u_c in place on the GPU.");
}
