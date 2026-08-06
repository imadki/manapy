#pragma once

// Precision configuration and nanobind array aliases shared by the
// manapy_compute_<float bits>_<int bits> binding translation units. Each target
// defines MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS to select the precisions.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

#include "array_view.hpp"
#include "precision.hpp"

#define MANAPY_COMPUTE_STR_(x) #x
#define MANAPY_COMPUTE_STR(x) MANAPY_COMPUTE_STR_(x)

namespace nb = nanobind;

using FVec = nb::ndarray<real_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FMat = nb::ndarray<real_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// 3D float array (e.g. to_convert.py's `cell_nf`, a per-cell array of
// per-face normals). No other kernel needs rank 3, so only this one alias.
using FTensor = nb::ndarray<real_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using CFVec =
    nb::ndarray<const real_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using CFMat =
    nb::ndarray<const real_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using IVec = nb::ndarray<index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IMat = nb::ndarray<index_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using CIVec =
    nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using CIMat =
    nb::ndarray<const index_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// int8 arrays (e.g. to_convert.py's `cell_type`, `b_visited`).
using I8Vec =
    nb::ndarray<std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using CI8Vec =
    nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Device (CuPy) counterparts of the aliases above. nanobind ingests them
// zero-copy through the DLPack protocol; their data pointers reference GPU
// memory, so make_view yields ArrayViews that a CUDA kernel can dereference
// directly.
using DFVec = nb::ndarray<real_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using DCFVec =
    nb::ndarray<const real_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using DCFMat =
    nb::ndarray<const real_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using DCIVec =
    nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using DCIMat =
    nb::ndarray<const index_t, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

// Construct an ArrayView from any nanobind ndarray at the binding boundary.
// T must match the ndarray's scalar type (const-qualified for const arrays),
// e.g. make_view<const real_t, 2>(cell_center).
template <typename T, int NDIM, typename... Args>
ArrayView<T, NDIM> make_view(const nb::ndarray<Args...> &arr) {
  ArrayView<T, NDIM> v;
  v.data = arr.data();
  for (int d = 0; d < NDIM; ++d) {
    v.shape[d] = static_cast<std::int64_t>(arr.shape(d));
    v.stride[d] = arr.stride(d);
  }
  return v;
}
