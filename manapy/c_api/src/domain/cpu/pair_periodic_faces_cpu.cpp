#include "domain_compute.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

// Unlike most kernels here, the per-side scratch (lo/hi/klo/khi below) is
// sized by a first counting pass and can scale with the whole mesh's
// boundary face count, so it isn't bounded by a small compile-time constant
// the way e.g. domain_helpers.hpp's get_phyid's max_face_nodes is.
// Heap-allocated with std::vector instead -- fine since this runs once per
// mesh build, not in a per-element hot loop.
index_t pair_periodic_faces(ArrayView<const index_t, 1> face_name,
                             ArrayView<const real_t, 2> face_center,
                             ArrayView<index_t, 2> face_cellid,
                             ArrayView<real_t, 2> cell_shift,
                             ArrayView<const real_t, 1> cmin, index_t name_lo,
                             index_t name_hi, index_t taxis0, index_t taxis1,
                             index_t saxis, real_t L, real_t dtol) {
  constexpr std::int64_t mult = 2147483648LL; // 2^31

  const index_t nf = static_cast<index_t>(face_name.size(0));
  index_t nlo = 0, nhi = 0;
  for (index_t i = 0; i < nf; ++i) {
    if (face_name(i) == name_lo)
      ++nlo;
    else if (face_name(i) == name_hi)
      ++nhi;
  }
  if (nlo == 0 && nhi == 0)
    return 0;
  if (nlo != nhi)
    return -1;

  std::vector<index_t> lo(nlo), hi(nhi);
  std::vector<std::int64_t> klo(nlo), khi(nhi);
  index_t a = 0, b = 0;
  for (index_t i = 0; i < nf; ++i) {
    if (face_name(i) != name_lo && face_name(i) != name_hi)
      continue;
    const std::int64_t k0 = static_cast<std::int64_t>(
        (face_center(i, taxis0) - cmin(taxis0)) / dtol + real_t(0.5));
    std::int64_t k1 = 0;
    if (taxis1 >= 0)
      k1 = static_cast<std::int64_t>(
          (face_center(i, taxis1) - cmin(taxis1)) / dtol + real_t(0.5));
    const std::int64_t key = k0 * mult + k1;
    if (face_name(i) == name_lo) {
      lo[a] = i;
      klo[a] = key;
      ++a;
    } else {
      hi[b] = i;
      khi[b] = key;
      ++b;
    }
  }

  std::vector<index_t> olo(nlo), ohi(nhi);
  std::iota(olo.begin(), olo.end(), index_t(0));
  std::iota(ohi.begin(), ohi.end(), index_t(0));
  std::sort(olo.begin(), olo.end(),
            [&](index_t x, index_t y) { return klo[x] < klo[y]; });
  std::sort(ohi.begin(), ohi.end(),
            [&](index_t x, index_t y) { return khi[x] < khi[y]; });

  for (index_t j = 0; j < nlo; ++j) {
    if (klo[olo[j]] != khi[ohi[j]])
      return -2;
    const index_t f = lo[olo[j]];
    const index_t fb = hi[ohi[j]];
    const index_t ca = face_cellid(f, 0);
    const index_t cb = face_cellid(fb, 0);
    face_cellid(f, 1) = cb;
    face_cellid(fb, 1) = ca;
    cell_shift(ca, saxis) = L;
    cell_shift(cb, saxis) = -L;
  }
  return nlo;
}
