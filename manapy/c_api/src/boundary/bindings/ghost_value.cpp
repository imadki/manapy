// Bindings for the scalar boundary conditions on local boundary faces
// (dirichlet / neumann / neumannNH / nonslip). This translation unit is
// compiled four times, once per manapy_compute_<float bits>_<int bits> package,
// with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

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

// The four conditions share one signature and differ only in which entry point
// they call, so one wrapper templated on that entry point serves them all. The
// first array is the Dirichlet `value` (per face) for ghost_value_dirichlet and
// the cell field `w_c` for the others; the m.def call names it accordingly.
// Same signature/argument order as the Python originals (minus start/stride,
// which the C++ loop supplies); w_ghost is written in place.
template <auto Fn>
void ghost_value_py(CFVec w, FVec w_ghost, CIMat face_cellid, CIVec bc_faces,
                    CFVec cst, CFVec face_dist_ortho) {
  Fn(make_view<const real_t, 1>(w), make_view<real_t, 1>(w_ghost),
     make_view<const index_t, 2>(face_cellid),
     make_view<const index_t, 1>(bc_faces), make_view<const real_t, 1>(cst),
     make_view<const real_t, 1>(face_dist_ortho));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per bc_faces entry, and w_ghost is written in place on the GPU.
template <auto Launch>
void ghost_value_cuda_py(DCFVec w, DFVec w_ghost, DCIMat face_cellid,
                         DCIVec bc_faces, DCFVec cst, DCFVec face_dist_ortho,
                         const char *what) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(w.device_id()), "cudaSetDevice");

  Launch(make_view<const real_t, 1>(w), make_view<real_t, 1>(w_ghost),
         make_view<const index_t, 2>(face_cellid),
         make_view<const index_t, 1>(bc_faces), make_view<const real_t, 1>(cst),
         make_view<const real_t, 1>(face_dist_ortho), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), what);
}

void ghost_value_dirichlet_cuda_py(DCFVec value, DFVec w_ghost,
                                   DCIMat face_cellid, DCIVec bc_faces,
                                   DCFVec cst, DCFVec face_dist_ortho) {
  ghost_value_cuda_py<&launch_ghost_value_dirichlet>(
      value, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho,
      "ghost_value_dirichlet kernel launch");
}

void ghost_value_neumann_cuda_py(DCFVec w_c, DFVec w_ghost, DCIMat face_cellid,
                                 DCIVec bc_faces, DCFVec cst,
                                 DCFVec face_dist_ortho) {
  ghost_value_cuda_py<&launch_ghost_value_neumann>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho,
      "ghost_value_neumann kernel launch");
}

void ghost_value_neumannNH_cuda_py(DCFVec w_c, DFVec w_ghost,
                                   DCIMat face_cellid, DCIVec bc_faces,
                                   DCFVec cst, DCFVec face_dist_ortho) {
  ghost_value_cuda_py<&launch_ghost_value_neumannNH>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho,
      "ghost_value_neumannNH kernel launch");
}

void ghost_value_nonslip_cuda_py(DCFVec w_c, DFVec w_ghost, DCIMat face_cellid,
                                 DCIVec bc_faces, DCFVec cst,
                                 DCFVec face_dist_ortho) {
  ghost_value_cuda_py<&launch_ghost_value_nonslip>(
      w_c, w_ghost, face_cellid, bc_faces, cst, face_dist_ortho,
      "ghost_value_nonslip kernel launch");
}

} // namespace

void register_ghost_value(nb::module_ &m) {
  m.def("ghost_value_dirichlet", &ghost_value_py<&ghost_value_dirichlet>,
        nb::arg("value"), nb::arg("w_ghost").noconvert(),
        nb::arg("face_cellid"), nb::arg("bc_faces"), nb::arg("cst"),
        nb::arg("face_dist_ortho"),
        "Dirichlet boundary condition on the faces listed in bc_faces: "
        "w_ghost[i] = value[i]. Writes w_ghost in place.");

  m.def("ghost_value_dirichlet_cuda", &ghost_value_dirichlet_cuda_py,
        nb::arg("value"), nb::arg("w_ghost").noconvert(),
        nb::arg("face_cellid"), nb::arg("bc_faces"), nb::arg("cst"),
        nb::arg("face_dist_ortho"),
        "GPU (CuPy) Dirichlet boundary condition. Takes device arrays and "
        "writes w_ghost in place on the GPU.");

  m.def("ghost_value_neumann", &ghost_value_py<&ghost_value_neumann>,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "Homogeneous Neumann (zero normal gradient) boundary condition on the "
        "faces listed in bc_faces: w_ghost[i] = w_c[face_cellid[i, 0]]. Writes "
        "w_ghost in place.");

  m.def("ghost_value_neumann_cuda", &ghost_value_neumann_cuda_py,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "GPU (CuPy) homogeneous Neumann boundary condition. Takes device "
        "arrays and writes w_ghost in place on the GPU.");

  m.def("ghost_value_neumannNH", &ghost_value_py<&ghost_value_neumannNH>,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "Inhomogeneous Neumann (imposed normal gradient cst) boundary "
        "condition on the faces listed in bc_faces: w_ghost[i] = "
        "w_c[face_cellid[i, 0]] + cst[i] * face_dist_ortho[i]. Writes w_ghost "
        "in place.");

  m.def("ghost_value_neumannNH_cuda", &ghost_value_neumannNH_cuda_py,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "GPU (CuPy) inhomogeneous Neumann boundary condition. Takes device "
        "arrays and writes w_ghost in place on the GPU.");

  m.def("ghost_value_nonslip", &ghost_value_py<&ghost_value_nonslip>,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "No-slip boundary condition on the faces listed in bc_faces: "
        "w_ghost[i] = -w_c[face_cellid[i, 0]], so the field vanishes at the "
        "wall. Writes w_ghost in place.");

  m.def("ghost_value_nonslip_cuda", &ghost_value_nonslip_cuda_py,
        nb::arg("w_c"), nb::arg("w_ghost").noconvert(), nb::arg("face_cellid"),
        nb::arg("bc_faces"), nb::arg("cst"), nb::arg("face_dist_ortho"),
        "GPU (CuPy) no-slip boundary condition. Takes device arrays and "
        "writes w_ghost in place on the GPU.");
}
