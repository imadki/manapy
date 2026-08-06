// Bindings for the diffusion solver's dissipative residual kernel. Compiled
// four times, once per manapy_compute_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h; keep before CUDA headers

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "bindings/registry.hpp"
#include "diffusion_compute.cuh"
#include "diffusion_compute.hpp"

namespace {

void cuda_check(cudaError_t err, const char *what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(err));
}

// Same signature/argument order as the Python original; dissip_w is written in
// place.
void explicitscheme_dissipative_py(CFVec wx_face, CFVec wy_face, CFVec wz_face,
                                   CIMat face_cellid, CFMat face_normal,
                                   CIVec face_name, FVec dissip_w, real_t Dxx,
                                   real_t Dyy, real_t Dzz) {
  explicitscheme_dissipative(
      make_view<const real_t, 1>(wx_face), make_view<const real_t, 1>(wy_face),
      make_view<const real_t, 1>(wz_face),
      make_view<const index_t, 2>(face_cellid),
      make_view<const real_t, 2>(face_normal),
      make_view<const index_t, 1>(face_name), make_view<real_t, 1>(dissip_w),
      Dxx, Dyy, Dzz);
}

// GPU version: device (CuPy) arrays; dissip_w written in place on the GPU.
void explicitscheme_dissipative_cuda_py(DCFVec wx_face, DCFVec wy_face,
                                        DCFVec wz_face, DCIMat face_cellid,
                                        DCFMat face_normal, DCIVec face_name,
                                        DFVec dissip_w, real_t Dxx, real_t Dyy,
                                        real_t Dzz) {
  cuda_check(cudaSetDevice(dissip_w.device_id()), "cudaSetDevice");

  launch_explicitscheme_dissipative(
      make_view<const real_t, 1>(wx_face), make_view<const real_t, 1>(wy_face),
      make_view<const real_t, 1>(wz_face),
      make_view<const index_t, 2>(face_cellid),
      make_view<const real_t, 2>(face_normal),
      make_view<const index_t, 1>(face_name), make_view<real_t, 1>(dissip_w),
      Dxx, Dyy, Dzz, /*stream=*/nullptr);

  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaDeviceSynchronize();
  cuda_check(err, "explicitscheme_dissipative kernel");
}

} // namespace

void register_diffusion_explicitscheme_dissipative(nb::module_ &m) {
  m.def("explicitscheme_dissipative", &explicitscheme_dissipative_py,
        nb::arg("wx_face"), nb::arg("wy_face"), nb::arg("wz_face"),
        nb::arg("face_cellid"), nb::arg("face_normal"), nb::arg("face_name"),
        nb::arg("dissip_w").noconvert(), nb::arg("Dxx"), nb::arg("Dyy"),
        nb::arg("Dzz"),
        "Diffusion (dissipative) residual: accumulates the anisotropic "
        "diffusion flux of each face into dissip_w (owner +q, interior "
        "neighbour -q). Writes dissip_w in place.");

  m.def("explicitscheme_dissipative_cuda", &explicitscheme_dissipative_cuda_py,
        nb::arg("wx_face"), nb::arg("wy_face"), nb::arg("wz_face"),
        nb::arg("face_cellid"), nb::arg("face_normal"), nb::arg("face_name"),
        nb::arg("dissip_w").noconvert(), nb::arg("Dxx"), nb::arg("Dyy"),
        nb::arg("Dzz"),
        "GPU (CuPy) diffusion (dissipative) residual. Takes device arrays and "
        "writes dissip_w in place on the GPU.");
}
