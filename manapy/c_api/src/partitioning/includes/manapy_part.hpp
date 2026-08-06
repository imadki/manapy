#pragma once

// Shared header for the partitioning module (METIS-backed domain
// decomposition, ported from the manapy c_api's includes/manapy_part.h).
//
// What the original header carried that is gone here:
//   - Python.h / numpy/arrayobject.h and the PyArray<T, Dim> wrapper, replaced
//     by ArrayView (inputs, src/base/array_view.hpp) and OwnedArray (outputs,
//     src/base/owned_array.hpp)
//   - Types.h's FLOAT_TYPE/INT_TYPE/NPY_*_TYPE macros, replaced by
//     src/base/precision.hpp's real_t / index_t
//   - the DEBUG_PRINT_INSTANT / DEBUG_TIME_IT macros, replaced by the
//     project-wide src/base/print_debug.hpp
//   - the compute_cell_center_* declarations, which now live with the rest of
//     the mesh geometry in src/domain/domain_compute.hpp

// owned_array.hpp / print_debug.hpp pull in Python.h, which has to be the
// first include in the translation unit: it selects the _POSIX_C_SOURCE and
// _XOPEN_SOURCE feature-test macros, and warns (then silently disagrees with
// itself) if a standard header got there first. metis.h includes inttypes.h,
// so it must come after these, not before.
#include "owned_array.hpp"
#include "precision.hpp"
#include "print_debug.hpp"

#include <metis.h>

#include <array>
#include <type_traits>
#include <vector>

#include "array_view.hpp"

// METIS is compiled once per precision pair precisely so these hold; see
// third_party/CMakeLists.txt. Without the first, every index array handed
// between this module and METIS would be silently reinterpreted; without the
// second, this header and metis.h declare incompatible global `real_t`s and
// the translation unit does not compile at all.
static_assert(std::is_same_v<idx_t, index_t>,
              "METIS idx_t is not index_t: IDXTYPEWIDTH disagrees with "
              "MANAPY_COMPUTE_INT_BITS");
static_assert(sizeof(real_t) * 8 == MANAPY_COMPUTE_FLOAT_BITS,
              "METIS real_t width disagrees with MANAPY_COMPUTE_FLOAT_BITS");

/* ---------------------------------------------------------------------- *
 *  Cell‑type enum – used to available geometry.                          *
 *-----------------------------------------------------------------------*/
enum CELL_TYPE {
    Triangle = 1,
    Quad = 2,
    Tetra = 3,
    Hexahedron = 4,
    Pyramid = 5
};

/* ---------------------------------------------------------------------- *
 *  utils.cpp
 *-----------------------------------------------------------------------*/

// Index of `item` within the valid prefix of `arr`, or -1. The prefix length
// is `arr`'s own last element, following the connectivity-row convention used
// throughout (a row stores its entry count in its final slot), so the search
// range is arr[0 .. arr[last]-1] and must be sorted ascending.
index_t binary_search(ArrayView<const index_t, 1> arr, index_t item);

// The (at most two) values common to every one of the `size` rows
// arr[indices(0)] .. arr[indices(size-1)]. Writes the hits into res[0], res[1]
// (res must hold at least 2 entries), leaving -1 in any slot not filled.
// Rows other than the first must be sorted ascending -- they are probed with
// binary_search.
void intersect_arr(ArrayView<const index_t, 2> arr,
                   ArrayView<const index_t, 1> indices, index_t size,
                   std::vector<index_t> &res);

// Per CELL_TYPE geometry limits: {max faces, max nodes per face, max nodes}.
// Returns all zeros for an unknown type.
std::array<index_t, 3> get_max_info(index_t cell_type);
