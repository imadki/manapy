// Bindings for the center-to-vertex kernel. This translation unit is compiled
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

// Same signature/argument order as the Python original; w_n is written in
// place. node_R_z / node_lambda_z are unused by the 2D kernel but kept for
// signature parity.
void center_to_vertex_2d_py(
    CFVec w_c, CFVec w_ghost, CFVec w_halo, CFVec w_haloghost, CFMat cell_center,
    CFMat halo_centvol, CIMat node_cellid, CFMat ghost_info_flt,
    CFMat ghost_ext_info_flt, CIMat node_ghostid, CIMat node_haloghostid,
    CIMat node_periodicid, CIMat node_halonid, CFMat nodes, CIVec node_oldname,
    CFVec node_R_x, CFVec node_R_y, CFVec node_R_z, CFVec node_lambda_x,
    CFVec node_lambda_y, CFVec node_lambda_z, CIVec node_number, CFMat cell_shift,
    FVec w_n, CIVec ghost_faceid) {
  (void)node_R_z;      // unused by the 2D kernel, kept for signature parity
  (void)node_lambda_z; // unused by the 2D kernel, kept for signature parity

  center_to_vertex_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(w_haloghost),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const index_t, 2>(node_cellid),
      make_view<const real_t, 2>(ghost_info_flt),
      make_view<const real_t, 2>(ghost_ext_info_flt),
      make_view<const index_t, 2>(node_ghostid),
      make_view<const index_t, 2>(node_haloghostid),
      make_view<const index_t, 2>(node_periodicid),
      make_view<const index_t, 2>(node_halonid),
      make_view<const real_t, 2>(nodes),
      make_view<const index_t, 1>(node_oldname),
      make_view<const real_t, 1>(node_R_x),
      make_view<const real_t, 1>(node_R_y),
      make_view<const real_t, 1>(node_lambda_x),
      make_view<const real_t, 1>(node_lambda_y),
      make_view<const index_t, 1>(node_number),
      make_view<const real_t, 2>(cell_shift), make_view<real_t, 1>(w_n),
      make_view<const index_t, 1>(ghost_faceid));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per node, and w_n is written in place on the GPU.
void center_to_vertex_2d_cuda_py(
    DCFVec w_c, DCFVec w_ghost, DCFVec w_halo, DCFVec w_haloghost,
    DCFMat cell_center, DCFMat halo_centvol, DCIMat node_cellid,
    DCFMat ghost_info_flt, DCFMat ghost_ext_info_flt, DCIMat node_ghostid,
    DCIMat node_haloghostid, DCIMat node_periodicid, DCIMat node_halonid,
    DCFMat nodes, DCIVec node_oldname, DCFVec node_R_x, DCFVec node_R_y,
    DCFVec node_R_z, DCFVec node_lambda_x, DCFVec node_lambda_y,
    DCFVec node_lambda_z, DCIVec node_number, DCFMat cell_shift, DFVec w_n,
    DCIVec ghost_faceid) {
  (void)node_R_z;      // unused by the 2D kernel, kept for signature parity
  (void)node_lambda_z; // unused by the 2D kernel, kept for signature parity

  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(w_c.device_id()), "cudaSetDevice");

  launch_center_to_vertex_2d(
      make_view<const real_t, 1>(w_c), make_view<const real_t, 1>(w_ghost),
      make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(w_haloghost),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const index_t, 2>(node_cellid),
      make_view<const real_t, 2>(ghost_info_flt),
      make_view<const real_t, 2>(ghost_ext_info_flt),
      make_view<const index_t, 2>(node_ghostid),
      make_view<const index_t, 2>(node_haloghostid),
      make_view<const index_t, 2>(node_periodicid),
      make_view<const index_t, 2>(node_halonid),
      make_view<const real_t, 2>(nodes),
      make_view<const index_t, 1>(node_oldname),
      make_view<const real_t, 1>(node_R_x),
      make_view<const real_t, 1>(node_R_y),
      make_view<const real_t, 1>(node_lambda_x),
      make_view<const real_t, 1>(node_lambda_y),
      make_view<const index_t, 1>(node_number),
      make_view<const real_t, 2>(cell_shift), make_view<real_t, 1>(w_n),
      make_view<const index_t, 1>(ghost_faceid), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), "center_to_vertex_2d kernel launch");
}

} // namespace

void register_center_to_vertex_2d(nb::module_ &m) {
  m.def(
      "center_to_vertex_2d", &center_to_vertex_2d_py, nb::arg("w_c"),
      nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_haloghost"),
      nb::arg("cell_center"), nb::arg("halo_centvol"), nb::arg("node_cellid"),
      nb::arg("ghost_info_flt"), nb::arg("ghost_ext_info_flt"),
      nb::arg("node_ghostid"), nb::arg("node_haloghostid"),
      nb::arg("node_periodicid"), nb::arg("node_halonid"), nb::arg("nodes"),
      nb::arg("node_oldname"), nb::arg("node_R_x"), nb::arg("node_R_y"),
      nb::arg("node_R_z"), nb::arg("node_lambda_x"), nb::arg("node_lambda_y"),
      nb::arg("node_lambda_z"), nb::arg("node_number"), nb::arg("cell_shift"),
      nb::arg("w_n").noconvert(), nb::arg("ghost_faceid"),
      "Distance-weighted interpolation of a cell field onto the vertices of a "
      "2D unstructured mesh. Writes the result into w_n in place.");

  m.def(
      "center_to_vertex_2d_cuda", &center_to_vertex_2d_cuda_py, nb::arg("w_c"),
      nb::arg("w_ghost"), nb::arg("w_halo"), nb::arg("w_haloghost"),
      nb::arg("cell_center"), nb::arg("halo_centvol"), nb::arg("node_cellid"),
      nb::arg("ghost_info_flt"), nb::arg("ghost_ext_info_flt"),
      nb::arg("node_ghostid"), nb::arg("node_haloghostid"),
      nb::arg("node_periodicid"), nb::arg("node_halonid"), nb::arg("nodes"),
      nb::arg("node_oldname"), nb::arg("node_R_x"), nb::arg("node_R_y"),
      nb::arg("node_R_z"), nb::arg("node_lambda_x"), nb::arg("node_lambda_y"),
      nb::arg("node_lambda_z"), nb::arg("node_number"), nb::arg("cell_shift"),
      nb::arg("w_n").noconvert(), nb::arg("ghost_faceid"),
      "GPU (CuPy) distance-weighted interpolation of a cell field onto the "
      "vertices of a 2D unstructured mesh. Takes device arrays and writes the "
      "result into w_n in place on the GPU.");
}
