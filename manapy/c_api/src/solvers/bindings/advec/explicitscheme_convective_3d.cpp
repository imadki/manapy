// Bindings for the 3D explicit convective residual kernel of the advection
// solver. This translation unit is compiled four times, once per
// manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "advec_compute.cuh"
#include "advec_compute.hpp"
#include "bindings/registry.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original; rez_w is written in
// place. All three velocity-gradient components (w_z / wz_halo included) are
// used in 3D.
void explicitscheme_convective_3d_py(
    FVec rez_w, CFVec w_c, CFVec w_ghost, CFVec w_halo, CFVec u_face,
    CFVec v_face, CFVec w_face, CFVec w_x, CFVec w_y, CFVec w_z, CFVec wx_halo,
    CFVec wy_halo, CFVec wz_halo, CFVec psi, CFVec psi_halo, CFMat cell_center,
    CFMat face_center, CFMat halo_centvol, CIMat face_cellid, CFMat face_normal,
    CIVec face_haloid, CIVec face_name, CIVec d_innerfaces, CIVec d_halofaces,
    CIVec d_boundaryfaces, CIVec d_periodicboundaryfaces, CFMat cell_shift,
    index_t order, index_t scheme) {
  explicitscheme_convective_3d(
      make_view<real_t, 1>(rez_w), make_view<const real_t, 1>(w_c),
      make_view<const real_t, 1>(w_ghost), make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(u_face), make_view<const real_t, 1>(v_face),
      make_view<const real_t, 1>(w_face), make_view<const real_t, 1>(w_x),
      make_view<const real_t, 1>(w_y), make_view<const real_t, 1>(w_z),
      make_view<const real_t, 1>(wx_halo), make_view<const real_t, 1>(wy_halo),
      make_view<const real_t, 1>(wz_halo), make_view<const real_t, 1>(psi),
      make_view<const real_t, 1>(psi_halo),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(face_center),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const index_t, 2>(face_cellid),
      make_view<const real_t, 2>(face_normal),
      make_view<const index_t, 1>(face_haloid),
      make_view<const index_t, 1>(face_name),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_halofaces),
      make_view<const index_t, 1>(d_boundaryfaces),
      make_view<const index_t, 1>(d_periodicboundaryfaces),
      make_view<const real_t, 2>(cell_shift), order, scheme);
}

// GPU version: same signature/argument order, but every array is a CuPy device
// array ingested zero-copy via DLPack. make_view builds ArrayViews straight
// over the device pointers and rez_w is written in place on the GPU.
void explicitscheme_convective_3d_cuda_py(
    DFVec rez_w, DCFVec w_c, DCFVec w_ghost, DCFVec w_halo, DCFVec u_face,
    DCFVec v_face, DCFVec w_face, DCFVec w_x, DCFVec w_y, DCFVec w_z,
    DCFVec wx_halo, DCFVec wy_halo, DCFVec wz_halo, DCFVec psi, DCFVec psi_halo,
    DCFMat cell_center, DCFMat face_center, DCFMat halo_centvol,
    DCIMat face_cellid, DCFMat face_normal, DCIVec face_haloid,
    DCIVec face_name, DCIVec d_innerfaces, DCIVec d_halofaces,
    DCIVec d_boundaryfaces, DCIVec d_periodicboundaryfaces, DCFMat cell_shift,
    index_t order, index_t scheme) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(rez_w.device_id()), "cudaSetDevice");

  launch_explicitscheme_convective_3d(
      make_view<real_t, 1>(rez_w), make_view<const real_t, 1>(w_c),
      make_view<const real_t, 1>(w_ghost), make_view<const real_t, 1>(w_halo),
      make_view<const real_t, 1>(u_face), make_view<const real_t, 1>(v_face),
      make_view<const real_t, 1>(w_face), make_view<const real_t, 1>(w_x),
      make_view<const real_t, 1>(w_y), make_view<const real_t, 1>(w_z),
      make_view<const real_t, 1>(wx_halo), make_view<const real_t, 1>(wy_halo),
      make_view<const real_t, 1>(wz_halo), make_view<const real_t, 1>(psi),
      make_view<const real_t, 1>(psi_halo),
      make_view<const real_t, 2>(cell_center),
      make_view<const real_t, 2>(face_center),
      make_view<const real_t, 2>(halo_centvol),
      make_view<const index_t, 2>(face_cellid),
      make_view<const real_t, 2>(face_normal),
      make_view<const index_t, 1>(face_haloid),
      make_view<const index_t, 1>(face_name),
      make_view<const index_t, 1>(d_innerfaces),
      make_view<const index_t, 1>(d_halofaces),
      make_view<const index_t, 1>(d_boundaryfaces),
      make_view<const index_t, 1>(d_periodicboundaryfaces),
      make_view<const real_t, 2>(cell_shift), order, scheme, /*stream=*/nullptr);

  // Surface launch errors, then block until the in-place writes are visible.
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "explicitscheme_convective_3d kernel");
}

} // namespace

void register_explicitscheme_convective_3d(nb::module_ &m) {
  m.def("explicitscheme_convective_3d", &explicitscheme_convective_3d_py,
        nb::arg("rez_w").noconvert(), nb::arg("w_c"), nb::arg("w_ghost"),
        nb::arg("w_halo"), nb::arg("u_face"), nb::arg("v_face"),
        nb::arg("w_face"), nb::arg("w_x"), nb::arg("w_y"), nb::arg("w_z"),
        nb::arg("wx_halo"), nb::arg("wy_halo"), nb::arg("wz_halo"),
        nb::arg("psi"), nb::arg("psi_halo"), nb::arg("cell_center"),
        nb::arg("face_center"), nb::arg("halo_centvol"), nb::arg("face_cellid"),
        nb::arg("face_normal"), nb::arg("face_haloid"), nb::arg("face_name"),
        nb::arg("d_innerfaces"), nb::arg("d_halofaces"),
        nb::arg("d_boundaryfaces"), nb::arg("d_periodicboundaryfaces"),
        nb::arg("cell_shift"), nb::arg("order"), nb::arg("scheme"),
        "Explicit finite-volume convective residual for 3D linear advection. "
        "Writes the per-cell residual into rez_w in place.");

  m.def("explicitscheme_convective_3d_cuda",
        &explicitscheme_convective_3d_cuda_py, nb::arg("rez_w").noconvert(),
        nb::arg("w_c"), nb::arg("w_ghost"), nb::arg("w_halo"),
        nb::arg("u_face"), nb::arg("v_face"), nb::arg("w_face"), nb::arg("w_x"),
        nb::arg("w_y"), nb::arg("w_z"), nb::arg("wx_halo"), nb::arg("wy_halo"),
        nb::arg("wz_halo"), nb::arg("psi"), nb::arg("psi_halo"),
        nb::arg("cell_center"), nb::arg("face_center"), nb::arg("halo_centvol"),
        nb::arg("face_cellid"), nb::arg("face_normal"), nb::arg("face_haloid"),
        nb::arg("face_name"), nb::arg("d_innerfaces"), nb::arg("d_halofaces"),
        nb::arg("d_boundaryfaces"), nb::arg("d_periodicboundaryfaces"),
        nb::arg("cell_shift"), nb::arg("order"), nb::arg("scheme"),
        "GPU (CuPy) explicit finite-volume convective residual for 3D linear "
        "advection. Takes device arrays and writes the per-cell residual into "
        "rez_w in place on the GPU.");
}
