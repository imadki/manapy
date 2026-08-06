// Bindings for the cell-to-face averaging kernel. This translation unit is
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

// Same signature/argument order as the Python original; u_face is written in
// place.
void celltoface_py(CFVec u_cell, FVec u_face, CFVec u_ghost, CFVec u_halo,
                   CIMat face_cellid, CIVec face_halofid, CIVec d_innerfaces,
                   CIVec d_boundaryfaces, CIVec d_halofaces) {
  celltoface(
      make_view<const real_t, 1>(u_cell), make_view<real_t, 1>(u_face),
      make_view<const real_t, 1>(u_ghost), make_view<const real_t, 1>(u_halo),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(face_halofid),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_boundaryfaces),
      make_view<const index_t, 1>(d_halofaces));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernels run one
// thread per face-list entry, and u_face is written in place on the GPU.
void celltoface_cuda_py(DCFVec u_cell, DFVec u_face, DCFVec u_ghost,
                        DCFVec u_halo, DCIMat face_cellid, DCIVec face_halofid,
                        DCIVec d_innerfaces, DCIVec d_boundaryfaces,
                        DCIVec d_halofaces) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(u_cell.device_id()), "cudaSetDevice");

  launch_celltoface(
      make_view<const real_t, 1>(u_cell), make_view<real_t, 1>(u_face),
      make_view<const real_t, 1>(u_ghost), make_view<const real_t, 1>(u_halo),
      make_view<const index_t, 2>(face_cellid),
      make_view<const index_t, 1>(face_halofid),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_boundaryfaces),
      make_view<const index_t, 1>(d_halofaces), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), "celltoface kernel launch");
}

} // namespace

void register_celltoface(nb::module_ &m) {
  m.def("celltoface", &celltoface_py, nb::arg("u_cell"),
        nb::arg("u_face").noconvert(), nb::arg("u_ghost"), nb::arg("u_halo"),
        nb::arg("face_cellid"), nb::arg("face_halofid"),
        nb::arg("d_innerfaces"), nb::arg("d_boundaryfaces"),
        nb::arg("d_halofaces"),
        "Cell-to-face averaging of a cell field onto faces on a 2D/3D "
        "unstructured mesh. Writes the result into u_face in place.");

  m.def("celltoface_cuda", &celltoface_cuda_py, nb::arg("u_cell"),
        nb::arg("u_face").noconvert(), nb::arg("u_ghost"), nb::arg("u_halo"),
        nb::arg("face_cellid"), nb::arg("face_halofid"),
        nb::arg("d_innerfaces"), nb::arg("d_boundaryfaces"),
        nb::arg("d_halofaces"),
        "GPU (CuPy) cell-to-face averaging of a cell field onto faces on a "
        "2D/3D unstructured mesh. Takes device arrays and writes the result "
        "into u_face in place on the GPU.");
}
