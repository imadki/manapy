#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Face-to-cell averaging for a single cell i: u_c(i) is the arithmetic mean
// of u_face over the cell's faces (translation of facetocell in
// to_convert.py). Shared verbatim by the CPU loop (cpu/facetocell_cpu.cpp)
// and the CUDA kernel (gpu/facetocell_cuda.cu). Writes u_c at index i in
// place.
//
// The last column of cell_faceid holds the number of valid entries in that
// row.
MANAPY_COMPUTE_HOST_DEVICE
void facetocell_cell(index_t i, ArrayView<const real_t, 1> u_face,
                      ArrayView<const index_t, 2> cell_faceid,
                      ArrayView<real_t, 1> u_c) {
  const index_t nf = cell_faceid(i, cell_faceid.size(1) - 1);

  real_t acc = real_t(0);
  for (index_t j = 0; j < nf; ++j) {
    acc += u_face(cell_faceid(i, j));
  }

  u_c(i) = acc / static_cast<real_t>(nf);
}
