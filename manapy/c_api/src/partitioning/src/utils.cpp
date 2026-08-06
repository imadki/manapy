// Partitioning helpers, ported from the manapy c_api's src/utils.cpp.
//
// The timing/printing half of that file (get_time, env_enabled,
// manapy_debug_timing_enabled, get_time_as_string, print_instant, time_it) is
// not here: it became the project-wide src/base/print_debug.hpp, so the other
// modules can trace too. What remains is the three mesh helpers, transcribed
// with PyArray<T, N> -> ArrayView<T, N> and idx_t -> index_t.

#include "manapy_part.hpp"

index_t binary_search(ArrayView<const index_t, 1> arr, const index_t item) {
    const index_t size = arr(arr.size(0) - 1);
    index_t left = 0;
    index_t right = size - 1;
    while (left <= right) {
        const index_t mid = (left + right) / 2;
        const index_t mid_val = arr(mid);
        if (mid_val == item) {
            return mid;
        } else if (mid_val < item) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

void intersect_arr(ArrayView<const index_t, 2> arr,
                   ArrayView<const index_t, 1> indices, const index_t size,
                   std::vector<index_t> &res) {
    index_t counter = 0;

    res[0] = -1;
    res[1] = -1;

    auto arr1 = arr.row(indices(0));
    for (index_t i = 0; i < arr1(arr1.size(0) - 1); i++) {
        res[counter] = arr1(i);
        for (index_t j = 1; j < size; j++) {
            auto arr2 = arr.row(indices(j));
            if (binary_search(arr2, arr1(i)) == -1){
                res[counter] = -1;
                break;
            }
        }
        if (res[counter] != -1)
            counter++;
        if (counter >= 2)
            break;
    }
}


std::array<index_t, 3> get_max_info(const index_t cell_type) {
    if (cell_type == CELL_TYPE::Triangle) {
        return {3, 2, 3};
    }
    else if (cell_type == CELL_TYPE::Quad) {
        return {4, 2, 4};
    }
    else if (cell_type == CELL_TYPE::Tetra) {
        return {4, 3, 4};
    }
    else if (cell_type == CELL_TYPE::Hexahedron) {
        return {6, 4, 8};
    }
    else if (cell_type == CELL_TYPE::Pyramid) {
        return {5, 4, 5};
    }
    return {0, 0, 0};
}
