#pragma once

// Batch 0 of src/domain/Steps.md: internal helpers shared by several domain
// kernels but never called from Python directly, so they carry no nanobind
// binding. CPU-only (see Steps.md) — plain `inline`, no
// MANAPY_COMPUTE_HOST_DEVICE/CUDA concerns.

#include <cstdint>
#include <cmath>
#include <stdexcept>

#include "array_view.hpp"
#include "precision.hpp"

// _is_in_array: index of `item` in `array`, or -1. The element count is
// stored at array[-1] (last slot), not array.size(0).
inline index_t is_in_array(ArrayView<const index_t, 1> array, index_t item) {
  const index_t n = array(array.size(0) - 1);
  for (index_t i = 0; i < n; ++i) {
    if (array(i) == item)
      return i;
  }
  return -1;
}

// _binary_search: `array[0:array[-1]]` must be sorted ascending.
inline index_t binary_search(ArrayView<const index_t, 1> array, index_t item) {
  const index_t size = array(array.size(0) - 1);
  index_t left = 0;
  index_t right = size - 1;

  while (left <= right) {
    const index_t mid = (left + right) / 2;
    const index_t mid_val = array(mid);

    if (mid_val == item)
      return mid;
    else if (mid_val < item)
      left = mid + 1;
    else
      right = mid - 1;
  }

  return -1;
}

// Sorts the first `n` elements of `data` ascending, in place. Faces have at
// most a handful of nodes (quad = 4), so insertion sort beats pulling in
// <algorithm> for this.
inline void insertion_sort(ArrayView<index_t, 1> data, index_t n) {
  for (index_t i = 1; i < n; ++i) {
    const index_t key = data(i);
    index_t j = i - 1;
    while (j >= 0 && data(j) > key) {
      data(j + 1) = data(j);
      --j;
    }
    data(j + 1) = key;
  }
}

// _intersect_nodes: common neighboring cells of every node in a face (at
// most 2). Renamed from `intersect_nodes` to avoid a clash with
// intersect_common below.
inline void intersect_face_nodes(ArrayView<const index_t, 1> face_nodes,
                                  index_t nb_nodes,
                                  ArrayView<const index_t, 2> node_cellid,
                                  ArrayView<index_t, 1> intersect_cell) {
  index_t index = 0;
  intersect_cell(0) = -1;
  intersect_cell(1) = -1;

  const auto cells = node_cellid.row(face_nodes(0));
  const index_t ncells = cells(cells.size(0) - 1);

  for (index_t i = 0; i < ncells; ++i) {
    intersect_cell(index) = cells(i);
    for (index_t j = 1; j < nb_nodes; ++j) {
      if (binary_search(node_cellid.row(face_nodes(j)), cells(i)) == -1) {
        intersect_cell(index) = -1;
        break;
      }
    }
    if (intersect_cell(index) != -1)
      ++index;
    if (index >= 2)
      return;
  }
}

// _create_cell_faces: faces of a single cell, keyed by its node list and
// cell_type (1=triangle, 2=rectangle, 3=tetrahedron, 4=hexahedron,
// 5=pyramid). out_faces gets each face's nodes; size_info(k) is the node
// count of face k, size_info(-1) the number of faces.
//
// Node layouts:
//   triangle : lines  [0,1] [1,2] [2,0]
//   rectangle: lines  [0,1] [1,2] [2,3] [3,0]
//   tet      : tris   [0,1,2] [0,1,3] [0,2,3] [1,2,3]
//   hex      : quads  [0,1,2,3] [0,1,4,5] [1,2,5,6] [2,3,6,7] [0,3,4,7] [4,5,6,7]
//   pyramid  : quad   [0,1,2,3]; tris [0,1,4] [1,2,4] [2,3,4] [0,3,4]
inline void create_cell_faces(ArrayView<const index_t, 1> nodes,
                               ArrayView<index_t, 2> out_faces,
                               ArrayView<index_t, 1> size_info,
                               index_t cell_type) {
  constexpr index_t triangle = 1;
  constexpr index_t rectangle = 2;
  constexpr index_t tetrahedron = 3;
  constexpr index_t hexahedron = 4;
  constexpr index_t pyramid = 5;

  const index_t last = size_info.size(0) - 1;

  if (cell_type == triangle) {
    out_faces(0, 0) = nodes(0);
    out_faces(0, 1) = nodes(1);
    size_info(0) = 2;

    out_faces(1, 0) = nodes(1);
    out_faces(1, 1) = nodes(2);
    size_info(1) = 2;

    out_faces(2, 0) = nodes(2);
    out_faces(2, 1) = nodes(0);
    size_info(2) = 2;

    size_info(last) = 3;
  } else if (cell_type == rectangle) {
    out_faces(0, 0) = nodes(0);
    out_faces(0, 1) = nodes(1);
    size_info(0) = 2;

    out_faces(1, 0) = nodes(1);
    out_faces(1, 1) = nodes(2);
    size_info(1) = 2;

    out_faces(2, 0) = nodes(2);
    out_faces(2, 1) = nodes(3);
    size_info(2) = 2;

    out_faces(3, 0) = nodes(3);
    out_faces(3, 1) = nodes(0);
    size_info(3) = 2;

    size_info(last) = 4;
  } else if (cell_type == tetrahedron) {
    out_faces(0, 0) = nodes(0);
    out_faces(0, 1) = nodes(1);
    out_faces(0, 2) = nodes(2);
    size_info(0) = 3;

    out_faces(1, 0) = nodes(0);
    out_faces(1, 1) = nodes(1);
    out_faces(1, 2) = nodes(3);
    size_info(1) = 3;

    out_faces(2, 0) = nodes(0);
    out_faces(2, 1) = nodes(2);
    out_faces(2, 2) = nodes(3);
    size_info(2) = 3;

    out_faces(3, 0) = nodes(1);
    out_faces(3, 1) = nodes(2);
    out_faces(3, 2) = nodes(3);
    size_info(3) = 3;

    size_info(last) = 4;
  } else if (cell_type == hexahedron) {
    out_faces(0, 0) = nodes(0);
    out_faces(0, 1) = nodes(1);
    out_faces(0, 2) = nodes(2);
    out_faces(0, 3) = nodes(3);
    size_info(0) = 4;

    out_faces(1, 0) = nodes(0);
    out_faces(1, 1) = nodes(1);
    out_faces(1, 2) = nodes(4);
    out_faces(1, 3) = nodes(5);
    size_info(1) = 4;

    out_faces(2, 0) = nodes(1);
    out_faces(2, 1) = nodes(2);
    out_faces(2, 2) = nodes(5);
    out_faces(2, 3) = nodes(6);
    size_info(2) = 4;

    out_faces(3, 0) = nodes(2);
    out_faces(3, 1) = nodes(3);
    out_faces(3, 2) = nodes(6);
    out_faces(3, 3) = nodes(7);
    size_info(3) = 4;

    out_faces(4, 0) = nodes(0);
    out_faces(4, 1) = nodes(3);
    out_faces(4, 2) = nodes(4);
    out_faces(4, 3) = nodes(7);
    size_info(4) = 4;

    out_faces(5, 0) = nodes(4);
    out_faces(5, 1) = nodes(5);
    out_faces(5, 2) = nodes(6);
    out_faces(5, 3) = nodes(7);
    size_info(5) = 4;

    size_info(last) = 6;
  } else if (cell_type == pyramid) {
    out_faces(0, 0) = nodes(0);
    out_faces(0, 1) = nodes(1);
    out_faces(0, 2) = nodes(2);
    out_faces(0, 3) = nodes(3);
    size_info(0) = 4;

    out_faces(1, 0) = nodes(0);
    out_faces(1, 1) = nodes(1);
    out_faces(1, 2) = nodes(4);
    size_info(1) = 3;

    out_faces(2, 0) = nodes(1);
    out_faces(2, 1) = nodes(2);
    out_faces(2, 2) = nodes(4);
    size_info(2) = 3;

    out_faces(3, 0) = nodes(2);
    out_faces(3, 1) = nodes(3);
    out_faces(3, 2) = nodes(4);
    size_info(3) = 3;

    out_faces(4, 0) = nodes(0);
    out_faces(4, 1) = nodes(3);
    out_faces(4, 2) = nodes(4);
    size_info(4) = 3;

    size_info(last) = 5;
  }
}

// _triangle_area_3d
inline real_t triangle_area_3d(ArrayView<const real_t, 1> a,
                                ArrayView<const real_t, 1> b,
                                ArrayView<const real_t, 1> c) {
  const real_t ux = b(0) - a(0), uy = b(1) - a(1), uz = b(2) - a(2);
  const real_t vx = c(0) - a(0), vy = c(1) - a(1), vz = c(2) - a(2);

  const real_t cross_x = uy * vz - uz * vy;
  const real_t cross_y = uz * vx - ux * vz;
  const real_t cross_z = ux * vy - uy * vx;

  return std::sqrt(cross_x * cross_x + cross_y * cross_y +
                    cross_z * cross_z) *
         real_t(0.5);
}

// _triangle_normal_3d. The Python original allocates and returns a new
// np.zeros(3) array; every helper here instead writes into a caller-owned
// ArrayView out-param (`normal`), matching the no-heap-allocation style the
// rest of the codebase uses at this layer.
inline void triangle_normal_3d(ArrayView<const real_t, 1> a,
                                ArrayView<const real_t, 1> b,
                                ArrayView<const real_t, 1> c,
                                ArrayView<real_t, 1> normal) {
  const real_t ux = b(0) - a(0), uy = b(1) - a(1), uz = b(2) - a(2);
  const real_t vx = c(0) - a(0), vy = c(1) - a(1), vz = c(2) - a(2);

  normal(0) = uy * vz - uz * vy;
  normal(1) = uz * vx - ux * vz;
  normal(2) = ux * vy - uy * vx;
}

// _get_phyid: physical-face id whose node set equals face_nodes, or -1.
// `phy_faces` is intentionally non-const: like the Python original (which
// sorts the numpy *view* `phy_faces[phyid][0:n]` in place), matching rows
// get sorted into ascending node order the first time they're checked, so
// repeat lookups for the same phyid can compare directly.
inline index_t get_phyid(ArrayView<index_t, 2> phy_faces,
                          ArrayView<const index_t, 1> face_nodes,
                          ArrayView<const index_t, 2> node_phyid) {
  constexpr index_t max_face_nodes = 8; // generous bound: quads have 4

  const index_t nb_nodes = face_nodes(face_nodes.size(0) - 1);

  index_t sorted_storage[max_face_nodes];
  ArrayView<index_t, 1> sorted_face_node;
  sorted_face_node.data = sorted_storage;
  sorted_face_node.shape[0] = max_face_nodes;
  sorted_face_node.stride[0] = 1;
  for (index_t i = 0; i < nb_nodes; ++i)
    sorted_face_node(i) = face_nodes(i);
  insertion_sort(sorted_face_node, nb_nodes);

  const index_t n = face_nodes(0); // select any node, choosing node 0
  const index_t count = node_phyid(n, node_phyid.size(1) - 1);

  for (index_t k = 0; k < count; ++k) {
    const index_t phyid = node_phyid(n, k);
    if (phy_faces(phyid, phy_faces.size(1) - 1) != nb_nodes)
      continue;

    auto phyid_nodes = phy_faces.row(phyid);
    insertion_sort(phyid_nodes, nb_nodes);

    bool all_equal = true;
    for (index_t i = 0; i < nb_nodes; ++i) {
      if (phyid_nodes(i) != sorted_face_node(i)) {
        all_equal = false;
        break;
      }
    }
    if (all_equal)
      return phyid;
  }
  return -1;
}

// _search_halo_cell: index into halo_haloext of the halo cell whose global
// `item` is the global index of a neighbor cell, searched among node_halo_cells' neighbors.
inline index_t search_halo_cell(ArrayView<const index_t, 1> node_halo_cells,
                                 ArrayView<const index_t, 2> halo_haloext,
                                 index_t item) {
  const index_t count = node_halo_cells(node_halo_cells.size(0) - 1);
  for (index_t i = 0; i < count; ++i) {
    const index_t n_halo_cell = node_halo_cells(i);
    if (halo_haloext(n_halo_cell, 0) == item)
      return n_halo_cell;
  }
  throw std::runtime_error(
      "search_halo_cell: item not found in halo_haloext of node_halo_cells");
}

// _intersect: entries appearing in every one of `array[indices[0..size)]`
// (up to intersect.size(0) results). `size` mirrors the Python int8 count of
// `indices` to search; `b_visited` is scratch, sized to array's node range.
// size must not exceed 127
inline void intersect_common(ArrayView<const index_t, 1> indices,
                              std::int8_t size,
                              ArrayView<const index_t, 2> array,
                              ArrayView<std::int8_t, 1> b_visited,
                              ArrayView<index_t, 1> intersect) {
  index_t counter = 0;
  const index_t limit = intersect.size(0);
  for (index_t i = 0; i < limit; ++i)
    intersect(i) = -1;

  for (std::int8_t i = 0; i < size; ++i) {
    const auto a = array.row(indices(i));
    const index_t count = a(a.size(0) - 1);
    for (index_t j = 0; j < count; ++j)
      b_visited(a(j)) = 0;
  }

  for (std::int8_t i = 0; i < size; ++i) {
    const auto a = array.row(indices(i));
    const index_t count = a(a.size(0) - 1);
    for (index_t j = 0; j < count; ++j) {
      const index_t node = a(j);
      b_visited(node) += 1;
      if (b_visited(node) == size) {
        intersect(counter) = node;
        ++counter;
        if (counter == limit)
          return;
      }
    }
  }
}
