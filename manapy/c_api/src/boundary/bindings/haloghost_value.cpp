// Bindings for the scalar boundary conditions on halo ghosts (dirichlet /
// neumann / neumannNH / nonslip). This translation unit is compiled four times,
// once per manapy_compute_<float bits>_<int bits> package, with
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

// The four conditions share one signature and differ only in which entry point
// they call, so one wrapper templated on that entry point serves them all. The
// first array is the prescribed per-halo-ghost value array for
// haloghost_value_dirichlet and the halo cell field `w_halo` for the others;
// the m.def call names it accordingly. Same signature/argument order as the
// Python originals (minus start/stride, which the C++ loop supplies);
// w_haloghost is written in place.
template <auto Fn>
void haloghost_value_py(CFVec w, FVec w_haloghost, CIMat node_haloghostid,
                        CIMat ghost_ext_info_int, CFMat ghost_ext_info_flt,
                        index_t BCindex, CIVec d_halonodes, CFVec cst) {
  Fn(make_view<const real_t, 1>(w), make_view<real_t, 1>(w_haloghost),
     make_view<const index_t, 2>(node_haloghostid),
     make_view<const index_t, 2>(ghost_ext_info_int),
     make_view<const real_t, 2>(ghost_ext_info_flt), BCindex,
     make_view<const index_t, 1>(d_halonodes), make_view<const real_t, 1>(cst));
}

// GPU version: same signature/argument order as the CPU original, but every
// array is a CuPy device array ingested zero-copy via DLPack. make_view builds
// ArrayViews straight over the device pointers, the CUDA kernel runs one thread
// per d_halonodes entry, and w_haloghost is written in place on the GPU.
template <auto Launch>
void haloghost_value_cuda_py(DCFVec w, DFVec w_haloghost,
                             DCIMat node_haloghostid, DCIMat ghost_ext_info_int,
                             DCFMat ghost_ext_info_flt, index_t BCindex,
                             DCIVec d_halonodes, DCFVec cst, const char *what) {
  // Run on the device the inputs live on, over CuPy's default (legacy) stream.
  cuda_check(cudaSetDevice(w.device_id()), "cudaSetDevice");

  Launch(make_view<const real_t, 1>(w), make_view<real_t, 1>(w_haloghost),
         make_view<const index_t, 2>(node_haloghostid),
         make_view<const index_t, 2>(ghost_ext_info_int),
         make_view<const real_t, 2>(ghost_ext_info_flt), BCindex,
         make_view<const index_t, 1>(d_halonodes),
         make_view<const real_t, 1>(cst), /*stream=*/nullptr);

  // Cheap, non-blocking: catches a bad launch config (grid/block dims,
  // invalid args) without waiting for the kernel to finish. The legacy
  // default stream already orders these kernels against the caller's CuPy
  // ops, so no sync is needed for correctness; see base/cuda_launch.hpp,
  // and set MANAPY_CUDA_SYNC=1 to restore a per-launch device sync.
  cuda_check(manapy_cuda_post_launch(), what);
}

void haloghost_value_dirichlet_cuda_py(DCFVec value_haloghost,
                                       DFVec w_haloghost,
                                       DCIMat node_haloghostid,
                                       DCIMat ghost_ext_info_int,
                                       DCFMat ghost_ext_info_flt,
                                       index_t BCindex, DCIVec d_halonodes,
                                       DCFVec cst) {
  haloghost_value_cuda_py<&launch_haloghost_value_dirichlet>(
      value_haloghost, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst,
      "haloghost_value_dirichlet kernel launch");
}

void haloghost_value_neumann_cuda_py(DCFVec w_halo, DFVec w_haloghost,
                                     DCIMat node_haloghostid,
                                     DCIMat ghost_ext_info_int,
                                     DCFMat ghost_ext_info_flt, index_t BCindex,
                                     DCIVec d_halonodes, DCFVec cst) {
  haloghost_value_cuda_py<&launch_haloghost_value_neumann>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst,
      "haloghost_value_neumann kernel launch");
}

void haloghost_value_neumannNH_cuda_py(DCFVec w_halo, DFVec w_haloghost,
                                       DCIMat node_haloghostid,
                                       DCIMat ghost_ext_info_int,
                                       DCFMat ghost_ext_info_flt,
                                       index_t BCindex, DCIVec d_halonodes,
                                       DCFVec cst) {
  haloghost_value_cuda_py<&launch_haloghost_value_neumannNH>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst,
      "haloghost_value_neumannNH kernel launch");
}

void haloghost_value_nonslip_cuda_py(DCFVec w_halo, DFVec w_haloghost,
                                     DCIMat node_haloghostid,
                                     DCIMat ghost_ext_info_int,
                                     DCFMat ghost_ext_info_flt, index_t BCindex,
                                     DCIVec d_halonodes, DCFVec cst) {
  haloghost_value_cuda_py<&launch_haloghost_value_nonslip>(
      w_halo, w_haloghost, node_haloghostid, ghost_ext_info_int,
      ghost_ext_info_flt, BCindex, d_halonodes, cst,
      "haloghost_value_nonslip kernel launch");
}

} // namespace

void register_haloghost_value(nb::module_ &m) {
  m.def("haloghost_value_dirichlet",
        &haloghost_value_py<&haloghost_value_dirichlet>,
        nb::arg("value_haloghost"), nb::arg("w_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"), nb::arg("cst"),
        "Dirichlet boundary condition on every halo ghost tagged BCindex that "
        "hangs off a node of d_halonodes: w_haloghost[g] = value_haloghost[g]. "
        "Note the first array is the prescribed per-halo-ghost value array, "
        "not the halo cell field the other conditions take. Writes w_haloghost "
        "in place.");

  m.def("haloghost_value_dirichlet_cuda", &haloghost_value_dirichlet_cuda_py,
        nb::arg("value_haloghost"), nb::arg("w_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"), nb::arg("cst"),
        "GPU (CuPy) Dirichlet boundary condition on halo ghosts. Takes device "
        "arrays and writes w_haloghost in place on the GPU.");

  m.def("haloghost_value_neumann",
        &haloghost_value_py<&haloghost_value_neumann>, nb::arg("w_halo"),
        nb::arg("w_haloghost").noconvert(), nb::arg("node_haloghostid"),
        nb::arg("ghost_ext_info_int"), nb::arg("ghost_ext_info_flt"),
        nb::arg("BCindex"), nb::arg("d_halonodes"), nb::arg("cst"),
        "Homogeneous Neumann (zero normal gradient) boundary condition on "
        "every halo ghost tagged BCindex that hangs off a node of "
        "d_halonodes: w_haloghost[g] = w_halo[ghost_ext_info_int[g, 0]]. "
        "Writes w_haloghost in place.");

  m.def("haloghost_value_neumann_cuda", &haloghost_value_neumann_cuda_py,
        nb::arg("w_halo"), nb::arg("w_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"), nb::arg("cst"),
        "GPU (CuPy) homogeneous Neumann boundary condition on halo ghosts. "
        "Takes device arrays and writes w_haloghost in place on the GPU.");

  m.def("haloghost_value_neumannNH",
        &haloghost_value_py<&haloghost_value_neumannNH>, nb::arg("w_halo"),
        nb::arg("w_haloghost").noconvert(), nb::arg("node_haloghostid"),
        nb::arg("ghost_ext_info_int"), nb::arg("ghost_ext_info_flt"),
        nb::arg("BCindex"), nb::arg("d_halonodes"), nb::arg("cst"),
        "Inhomogeneous Neumann (imposed normal gradient cst) boundary "
        "condition on every halo ghost tagged BCindex that hangs off a node of "
        "d_halonodes: w_haloghost[g] = w_halo[ghost_ext_info_int[g, 0]] + "
        "cst[g] * 2*|ghost_ext_info_flt[g, 0]|. cst is indexed per halo ghost. "
        "Writes w_haloghost in place.");

  m.def("haloghost_value_neumannNH_cuda", &haloghost_value_neumannNH_cuda_py,
        nb::arg("w_halo"), nb::arg("w_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"), nb::arg("cst"),
        "GPU (CuPy) inhomogeneous Neumann boundary condition on halo ghosts. "
        "Takes device arrays and writes w_haloghost in place on the GPU.");

  m.def("haloghost_value_nonslip",
        &haloghost_value_py<&haloghost_value_nonslip>, nb::arg("w_halo"),
        nb::arg("w_haloghost").noconvert(), nb::arg("node_haloghostid"),
        nb::arg("ghost_ext_info_int"), nb::arg("ghost_ext_info_flt"),
        nb::arg("BCindex"), nb::arg("d_halonodes"), nb::arg("cst"),
        "No-slip boundary condition on every halo ghost tagged BCindex that "
        "hangs off a node of d_halonodes: w_haloghost[g] = "
        "-w_halo[ghost_ext_info_int[g, 0]], so the field vanishes at the wall. "
        "Writes w_haloghost in place.");

  m.def("haloghost_value_nonslip_cuda", &haloghost_value_nonslip_cuda_py,
        nb::arg("w_halo"), nb::arg("w_haloghost").noconvert(),
        nb::arg("node_haloghostid"), nb::arg("ghost_ext_info_int"),
        nb::arg("ghost_ext_info_flt"), nb::arg("BCindex"),
        nb::arg("d_halonodes"), nb::arg("cst"),
        "GPU (CuPy) no-slip boundary condition on halo ghosts. Takes device "
        "arrays and writes w_haloghost in place on the GPU.");
}
