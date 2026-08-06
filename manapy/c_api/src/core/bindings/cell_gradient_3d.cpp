// Bindings for the 3D cell-gradient kernel. This translation unit is compiled
// four times, once per manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

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

// Same signature/argument order as the Python original; w_x/w_y/w_z are
// written in place.
void cell_gradient_3d_py(CFVec w_c, CFVec w_ghost, CFVec w_halo,
                         CFVec w_haloghost, CFMat cell_center,
                         CIMat cell_cellnid, CFMat ghost_info_flt,
                         CFMat ghost_ext_info_flt, CIMat cell_ghostnid,
                         CIMat cell_haloghostnid, CIMat cell_halonid,
                         CIMat cells, CIMat cell_periodicfid,
                         CIMat node_periodicid, CIVec node_oldname,
                         CFMat halo_centvol, CFMat cell_shift, FVec w_x,
                         FVec w_y, FVec w_z, CIVec ghost_faceid) {
  (void)cells;           // unused by the kernel, kept for signature parity
  (void)node_periodicid; // unused by the kernel, kept for signature parity
  (void)node_oldname;    // unused by the kernel, kept for signature parity

  cell_gradient_3d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(w_haloghost),
      make_view<const real_t, 2>(cell_center),
      make_view<const index_t, 2>(cell_cellnid),
      make_view<const real_t, 2>(ghost_info_flt),
      make_view<const real_t, 2>(ghost_ext_info_flt),
      make_view<const index_t, 2>(cell_ghostnid),
      make_view<const index_t, 2>(cell_haloghostnid),
      make_view<const index_t, 2>(cell_halonid),
      make_view<const index_t, 2>(cell_periodicfid),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const real_t, 2>(cell_shift), make_view<real_t, 1>(w_x),
      make_view<real_t, 1>(w_y), make_view<real_t, 1>(w_z),
      make_view<const index_t, 1>(ghost_faceid));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per cell, and w_x/w_y/w_z are written in place on the GPU.
void cell_gradient_3d_cuda_py(
    DCFVec w_c, DCFVec w_ghost, DCFVec w_halo, DCFVec w_haloghost,
    DCFMat cell_center, DCIMat cell_cellnid, DCFMat ghost_info_flt,
    DCFMat ghost_ext_info_flt, DCIMat cell_ghostnid, DCIMat cell_haloghostnid,
    DCIMat cell_halonid, DCIMat cells, DCIMat cell_periodicfid,
    DCIMat node_periodicid, DCIVec node_oldname, DCFMat halo_centvol,
    DCFMat cell_shift, DFVec w_x, DFVec w_y, DFVec w_z, DCIVec ghost_faceid) {
  (void)cells;           // unused by the kernel, kept for signature parity
  (void)node_periodicid; // unused by the kernel, kept for signature parity
  (void)node_oldname;    // unused by the kernel, kept for signature parity

  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(w_c.device_id()), "cudaSetDevice");

  launch_cell_gradient_3d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(w_haloghost),
      make_view<const real_t, 2>(cell_center),
      make_view<const index_t, 2>(cell_cellnid),
      make_view<const real_t, 2>(ghost_info_flt),
      make_view<const real_t, 2>(ghost_ext_info_flt),
      make_view<const index_t, 2>(cell_ghostnid),
      make_view<const index_t, 2>(cell_haloghostnid),
      make_view<const index_t, 2>(cell_halonid),
      make_view<const index_t, 2>(cell_periodicfid),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const real_t, 2>(cell_shift), make_view<real_t, 1>(w_x),
      make_view<real_t, 1>(w_y), make_view<real_t, 1>(w_z),
      make_view<const index_t, 1>(ghost_faceid), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), "cell_gradient_3d kernel launch");
}

} // namespace

void register_cell_gradient_3d(nb::module_ &m) {
  m.def("cell_gradient_3d", &cell_gradient_3d_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_haloghost"),
        nb::arg("cell_center"), nb::arg("cell_cellnid"),
        nb::arg("ghost_info_flt"), nb::arg("ghost_ext_info_flt"),
        nb::arg("cell_ghostnid"), nb::arg("cell_haloghostnid"),
        nb::arg("cell_halonid"), nb::arg("cells"), nb::arg("cell_periodicfid"),
        nb::arg("node_periodicid"), nb::arg("node_oldname"),
        nb::arg("halo_centvol"), nb::arg("cell_shift"),
        nb::arg("w_x").noconvert(), nb::arg("w_y").noconvert(),
        nb::arg("w_z").noconvert(), nb::arg("ghost_faceid"),
        "Least-squares gradient of a cell field on a 3D unstructured mesh. "
        "Writes the result into w_x, w_y, w_z in place.");

  m.def("cell_gradient_3d_cuda", &cell_gradient_3d_cuda_py, nb::arg("w_c"),
        nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_haloghost"),
        nb::arg("cell_center"), nb::arg("cell_cellnid"),
        nb::arg("ghost_info_flt"), nb::arg("ghost_ext_info_flt"),
        nb::arg("cell_ghostnid"), nb::arg("cell_haloghostnid"),
        nb::arg("cell_halonid"), nb::arg("cells"), nb::arg("cell_periodicfid"),
        nb::arg("node_periodicid"), nb::arg("node_oldname"),
        nb::arg("halo_centvol"), nb::arg("cell_shift"),
        nb::arg("w_x").noconvert(), nb::arg("w_y").noconvert(),
        nb::arg("w_z").noconvert(), nb::arg("ghost_faceid"),
        "GPU (CuPy) least-squares gradient of a cell field on a 3D "
        "unstructured mesh. Takes device arrays and writes the result into "
        "w_x, w_y, w_z in place on the GPU.");
}
