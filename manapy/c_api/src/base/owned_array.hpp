#pragma once

// OwnedArray: a dense, C-contiguous array that owns its buffer on the C++ side
// and can hand that buffer to Python without copying it.
//
// Most kernels in this project follow the "caller preallocates, kernel fills in
// place" convention -- the binding takes an nb::ndarray, make_view() turns it
// into an ArrayView, and nothing is allocated on the C++ side. That works
// because the caller knows every output size up front.
//
// src/partitioning is the exception: sizes like the halo count or the number of
// physical faces owned by a partition are only known once the partitioning has
// run, so those kernels must allocate and return. OwnedArray is the allocation
// half of that; release() is the handoff. It replaces the NumPy-C-API-backed
// PyArray<T, Dim> of the original manapy c_api:
//
//     PyArray_SimpleNew(...)                 ->  OwnedArray<T, N>(shape)
//     PyArray_ZEROS(...)                     ->  OwnedArray<T, N>(shape, true)
//     PyArray_SimpleNewFromData + OWNDATA    ->  release()
//     arr->get(i) / arr->get2(i, j)          ->  arr(i) / arr(i, j)
//     arr->shape[d]                          ->  arr.size(d)
//     arr->ref_holder (handed to Py_BuildValue)  ->  arr.release()
//     arr == nullptr                         ->  !arr
//     delete arr; arr = nullptr;             ->  arr.reset()

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "array_view.hpp"

namespace nb = nanobind;

// Build a shape from a variadic list of integers, so allocation sites read the
// same way they did with the original make_npy_dims():
//   OwnedArray<index_t, 2> halos(make_dims(nb_halos, max_nodeid + 2));
template <typename... Ints>
constexpr std::array<std::size_t, sizeof...(Ints)> make_dims(Ints... ns) {
  return {static_cast<std::size_t>(ns)...};
}

template <typename T, int NDIM>
class OwnedArray {
  static_assert(NDIM >= 1 && NDIM <= 3, "OwnedArray supports rank 1 to 3");

public:
  // Empty: no buffer, zero extents. This is where a default-constructed table
  // starts, what release() leaves behind, and what reset() returns it to -- so
  // "holds a buffer" is a single notion, testable with `if (arr)`. It exists
  // so a struct of not-yet-allocated tables (LocalDomainStruct) can hold them
  // by value rather than wrapping each in an optional, which would add a
  // second, independent emptiness flag that could disagree with this one.
  OwnedArray() = default;

  // Uninitialized (PyArray_SimpleNew) or zero-filled (PyArray_ZEROS). Zeroing
  // is not the default: several kernels write every element and would only pay
  // for the memset. Kernels that accumulate into their output -- e.g. the
  // per-partition `nodes` table -- must pass zero_init = true.
  explicit OwnedArray(std::array<std::size_t, NDIM> shape,
                      bool zero_init = false)
      : shape_(shape) {
    std::size_t n = 1;
    for (int d = 0; d < NDIM; ++d) {
      // Guard the shape product: these sizes are derived from mesh data, and a
      // silent wraparound here would under-allocate rather than fail.
      if (shape_[d] != 0 && n > SIZE_MAX / shape_[d])
        throw std::bad_alloc();
      n *= shape_[d];
    }
    count_ = n;
    // new T[n]() value-initializes (zeroes) the buffer; new T[n] leaves it
    // uninitialized. Both throw std::bad_alloc on failure, which nanobind
    // surfaces to Python as MemoryError.
    buf_.reset(zero_init ? new T[n]() : new T[n]);
  }

  OwnedArray(const OwnedArray &) = delete;
  OwnedArray &operator=(const OwnedArray &) = delete;
  OwnedArray(OwnedArray &&) = default;
  OwnedArray &operator=(OwnedArray &&) = default;

  // Element access, mirroring ArrayView's call syntax so ported code reads the
  // same whether it is working on an input view or on an owned output.
  T &operator()(std::int64_t i) const
    requires(NDIM == 1)
  {
    return buf_[i];
  }

  T &operator()(std::int64_t i, std::int64_t j) const
    requires(NDIM == 2)
  {
    return buf_[i * static_cast<std::int64_t>(shape_[1]) + j];
  }

  T &operator()(std::int64_t i, std::int64_t j, std::int64_t k) const
    requires(NDIM == 3)
  {
    return buf_[(i * static_cast<std::int64_t>(shape_[1]) + j) *
                    static_cast<std::int64_t>(shape_[2]) +
                k];
  }

  std::int64_t size(int dim) const {
    return static_cast<std::int64_t>(shape_[dim]);
  }

  // Total element count, i.e. the product of all extents.
  std::size_t count() const { return count_; }

  T *data() const { return buf_.get(); }

  // Whether this array holds a buffer. False for a default-constructed one,
  // and for one whose buffer has been release()d or reset().
  explicit operator bool() const { return static_cast<bool>(buf_); }

  // Drop the buffer and return to the empty state -- the by-value counterpart
  // of `delete arr; arr = nullptr;`.
  void reset() {
    buf_.reset();
    shape_.fill(0);
    count_ = 0;
  }

  // Non-owning view, for handing an owned output to a routine written against
  // ArrayView (everything in src/base and the shared kernel code).
  ArrayView<T, NDIM> view() const {
    ArrayView<T, NDIM> v;
    v.data = buf_.get();
    std::int64_t stride = 1;
    for (int d = NDIM - 1; d >= 0; --d) {
      v.shape[d] = static_cast<std::int64_t>(shape_[d]);
      v.stride[d] = stride;
      stride *= static_cast<std::int64_t>(shape_[d]);
    }
    return v;
  }

  // Hand the buffer to Python. Ownership moves to the returned array: a capsule
  // holding the pointer delete[]s it once NumPy drops its last reference, which
  // is the nanobind equivalent of PyArray_SimpleNewFromData followed by
  // PyArray_ENABLEFLAGS(NPY_ARRAY_OWNDATA). No copy is made, and this
  // OwnedArray is empty afterwards.
  nb::ndarray<nb::numpy, T, nb::ndim<NDIM>, nb::c_contig> release() {
    if (!buf_)
      throw std::logic_error("OwnedArray::release() on an empty array");

    T *p = buf_.release();
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<T *>(q); });
    // ndarray_create copies the shape, so pointing at the member is fine --
    // and so is clearing it immediately afterwards.
    nb::ndarray<nb::numpy, T, nb::ndim<NDIM>, nb::c_contig> out(
        p, NDIM, shape_.data(), owner);
    shape_.fill(0);
    count_ = 0;
    return out;
  }

private:
  std::unique_ptr<T[]> buf_;
  std::array<std::size_t, NDIM> shape_{};
  std::size_t count_ = 0;
};
