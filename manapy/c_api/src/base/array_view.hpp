#pragma once

#include <cstdint>

// __host__ __device__ only means something to nvcc; plain C++ translation
// units (e.g. the gradient bindings) see a regular inline function.
#if defined(__CUDACC__)
#define MANAPY_COMPUTE_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define MANAPY_COMPUTE_HOST_DEVICE inline
#endif

// Non-owning view of a dense (possibly strided) array, usable on both host
// and device. Strides are in elements, not bytes.
template <typename T, int NDIM>
struct ArrayView {
  T *data;
  std::int64_t shape[NDIM];
  std::int64_t stride[NDIM]; // in elements, not bytes

  MANAPY_COMPUTE_HOST_DEVICE
  T &operator()(std::int64_t i) const
    requires(NDIM == 1)
  {
    return data[i * stride[0]];
  }

  MANAPY_COMPUTE_HOST_DEVICE
  T &operator()(std::int64_t i, std::int64_t j) const
    requires(NDIM == 2)
  {
    return data[i * stride[0] + j * stride[1]];
  }

  MANAPY_COMPUTE_HOST_DEVICE
  T &operator()(std::int64_t i, std::int64_t j, std::int64_t k) const
    requires(NDIM == 3)
  {
    return data[i * stride[0] + j * stride[1] + k * stride[2]];
  }

  MANAPY_COMPUTE_HOST_DEVICE
  std::int64_t size(int dim) const { return shape[dim]; }

  // The final slot of a 1D view. Connectivity rows throughout this project
  // store their own entry count there (`array[-1]` in the Python original), so
  // `row(i).last()` is how many of that row's entries are valid. Assignable,
  // since the count has to be written when a row is built.
  MANAPY_COMPUTE_HOST_DEVICE
  T &last() const
    requires(NDIM == 1)
  {
    return (*this)(shape[0] - 1);
  }

  // View row `i` as a 1D array, respecting its column stride. Lets code
  // written against `ArrayView<T, 1>` (e.g. domain_helpers.hpp's
  // binary_search, insertion_sort) run directly on one row of a 2D array
  // (e.g. node_cellid(node)) without copying it out.
  MANAPY_COMPUTE_HOST_DEVICE
  ArrayView<T, 1> row(std::int64_t i) const
    requires(NDIM == 2)
  {
    ArrayView<T, 1> v;
    v.data = &(*this)(i, 0);
    v.shape[0] = shape[1];
    v.stride[0] = stride[1];
    return v;
  }

  // Read-only view over the same data. Lets a mutable scratch buffer (e.g.
  // a `row()` of one) be passed to a function that only reads its argument
  // and says so in its signature (`ArrayView<const T, NDIM>`).
  MANAPY_COMPUTE_HOST_DEVICE
  ArrayView<const T, NDIM> as_const() const {
    ArrayView<const T, NDIM> v;
    v.data = data;
    for (int d = 0; d < NDIM; ++d) {
      v.shape[d] = shape[d];
      v.stride[d] = stride[d];
    }
    return v;
  }
};
