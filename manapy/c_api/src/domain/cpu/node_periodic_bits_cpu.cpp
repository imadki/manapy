#include "domain_compute.hpp"

void node_periodic_bits(ArrayView<const index_t, 2> faces,
                         ArrayView<const index_t, 1> face_name,
                         ArrayView<index_t, 1> node_bits) {
  const index_t nf = static_cast<index_t>(face_name.size(0));
  const index_t faces_last = static_cast<index_t>(faces.size(1)) - 1;
  for (index_t f = 0; f < nf; ++f) {
    const index_t nm = face_name(f);
    index_t bit = 0;
    if (nm == 11)
      bit = 1;
    else if (nm == 22)
      bit = 2;
    else if (nm == 33)
      bit = 4;
    else if (nm == 44)
      bit = 8;
    else if (nm == 55)
      bit = 16;
    else if (nm == 66)
      bit = 32;
    if (bit == 0)
      continue;
    const index_t count = faces(f, faces_last);
    for (index_t j = 0; j < count; ++j) {
      const index_t nd = faces(f, j);
      node_bits(nd) = node_bits(nd) | bit;
    }
  }
}
