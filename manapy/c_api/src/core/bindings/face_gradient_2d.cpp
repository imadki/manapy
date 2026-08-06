// Bindings for the face-gradient kernel. This translation unit is compiled
// four times, once per manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

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

// Same signature/argument order as the Python original; wx_face/wy_face are
// written in place.
void face_gradient_2d_py(CFVec w_c, CFVec w_ghost, CFVec w_halo, CFVec w_node,
                         CIMat face_cellid, CIMat faces, CIVec face_halofid,
                         CFVec face_airDiamond, CFMat face_normal, CFMat face_f1,
                         CFMat face_f2, CFMat face_f3, CFMat face_f4,
                         FVec wx_face, FVec wy_face, FVec wz_face,
                         CIVec d_innerfaces, CIVec d_halofaces,
                         CIVec dirichletfaces, CIVec neumann,
                         CIVec d_periodicfaces) {
  (void)face_normal; // unused by the kernel, kept for signature parity
  (void)wz_face;     // unused by the kernel, kept for signature parity

  face_gradient_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo), make_view<const real_t, 1>(w_node),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 2>(faces),
      make_view<const index_t, 1>(face_halofid),
      make_view<const real_t, 1>(face_airDiamond),
      make_view<const real_t, 2>(face_f1), make_view<const real_t, 2>(face_f2),
      make_view<const real_t, 2>(face_f3), make_view<const real_t, 2>(face_f4),
      make_view<real_t, 1>(wx_face), make_view<real_t, 1>(wy_face),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_halofaces),
      make_view<const index_t, 1>(dirichletfaces),
      make_view<const index_t, 1>(neumann),
      make_view<const index_t, 1>(d_periodicfaces));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernels run one
// thread per face-list entry, and wx_face/wy_face are written in place on the
// GPU.
void face_gradient_2d_cuda_py(
    DCFVec w_c, DCFVec w_ghost, DCFVec w_halo, DCFVec w_node,
    DCIMat face_cellid, DCIMat faces, DCIVec face_halofid,
    DCFVec face_airDiamond, DCFMat face_normal, DCFMat face_f1, DCFMat face_f2,
    DCFMat face_f3, DCFMat face_f4, DFVec wx_face, DFVec wy_face,
    DFVec wz_face, DCIVec d_innerfaces, DCIVec d_halofaces,
    DCIVec dirichletfaces, DCIVec neumann, DCIVec d_periodicfaces,
    uintptr_t stream_ptr) {
  (void)face_normal; // unused by the kernel, kept for signature parity
  (void)wz_face;     // unused by the kernel, kept for signature parity

  // Run on the device the inputs live on.
  cuda_check(cudaSetDevice(w_c.device_id()), "cudaSetDevice");

  // stream_ptr is a raw cudaStream_t handle (e.g. cupy.cuda.Stream.ptr) so the
  // caller picks which stream the kernels run on; 0 is the default/legacy
  // stream. This call is asynchronous: it returns as soon as the kernels are
  // enqueued, so the caller controls when (and whether) to synchronize.
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  launch_face_gradient_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo), make_view<const real_t, 1>(w_node),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 2>(faces),
      make_view<const index_t, 1>(face_halofid),
      make_view<const real_t, 1>(face_airDiamond),
      make_view<const real_t, 2>(face_f1), make_view<const real_t, 2>(face_f2),
      make_view<const real_t, 2>(face_f3), make_view<const real_t, 2>(face_f4),
      make_view<real_t, 1>(wx_face), make_view<real_t, 1>(wy_face),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_halofaces),
      make_view<const index_t, 1>(dirichletfaces),
      make_view<const index_t, 1>(neumann),
      make_view<const index_t, 1>(d_periodicfaces), stream);

  // Cheap, non-blocking: catches bad launch config (grid/block dims, invalid
  // args) immediately without waiting for kernel completion. cudaGetLastError
  // (not cudaPeekAtLastError) so the sticky CUDA error state is cleared here
  // rather than bleeding into whatever unrelated call checks it next.
  cuda_check(cudaGetLastError(), "face_gradient_2d kernel launch");
}

} // namespace

void register_face_gradient_2d(nb::module_ &m) {
  m.def("face_gradient_2d", &face_gradient_2d_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_node"),
        nb::arg("face_cellid"), nb::arg("faces"), nb::arg("face_halofid"),
        nb::arg("face_airDiamond"), nb::arg("face_normal"), nb::arg("face_f1"),
        nb::arg("face_f2"), nb::arg("face_f3"), nb::arg("face_f4"),
        nb::arg("wx_face").noconvert(), nb::arg("wy_face").noconvert(),
        nb::arg("wz_face").noconvert(), nb::arg("d_innerfaces"),
        nb::arg("d_halofaces"), nb::arg("dirichletfaces"), nb::arg("neumann"),
        nb::arg("d_periodicfaces"),
        "Green-Gauss-style face gradient of a cell field on a 2D unstructured "
        "mesh. Writes the result into wx_face, wy_face in place.");

  m.def("face_gradient_2d_cuda", &face_gradient_2d_cuda_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_node"),
        nb::arg("face_cellid"), nb::arg("faces"), nb::arg("face_halofid"),
        nb::arg("face_airDiamond"), nb::arg("face_normal"), nb::arg("face_f1"),
        nb::arg("face_f2"), nb::arg("face_f3"), nb::arg("face_f4"),
        nb::arg("wx_face").noconvert(), nb::arg("wy_face").noconvert(),
        nb::arg("wz_face").noconvert(), nb::arg("d_innerfaces"),
        nb::arg("d_halofaces"), nb::arg("dirichletfaces"), nb::arg("neumann"),
        nb::arg("d_periodicfaces"), nb::arg("stream_ptr") = 0,
        "GPU (CuPy) Green-Gauss-style face gradient of a cell field on a 2D "
        "unstructured mesh. Takes device arrays and writes the result into "
        "wx_face, wy_face in place on the GPU. Asynchronous: launches on the "
        "stream given by stream_ptr (a raw cudaStream_t handle, e.g. "
        "cupy.cuda.Stream.ptr; 0 is the default stream) and returns without "
        "synchronizing, so the caller controls when to synchronize.");
}
