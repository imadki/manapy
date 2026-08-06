#include "domain_compute.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

// Same heap-scratch rationale as pair_periodic_faces_cpu.cpp: the per-side
// scratch scales with the mesh's boundary node count, so it's
// std::vector-backed rather than a fixed-size stack buffer.
void accum_periodic_dir(ArrayView<const index_t, 1> node_bits,
                         ArrayView<const real_t, 2> nodes,
                         ArrayView<const index_t, 2> node_cellid,
                         ArrayView<index_t, 2> node_periodicid,
                         ArrayView<index_t, 1> node_fill,
                         ArrayView<const real_t, 1> cmin, index_t lo_bit,
                         index_t hi_bit, index_t taxis0, index_t taxis1,
                         real_t dtol) {
  constexpr std::int64_t mult = 2147483648LL;

  const index_t nn = static_cast<index_t>(node_bits.size(0));
  index_t nlo = 0, nhi = 0;
  for (index_t i = 0; i < nn; ++i) {
    if ((node_bits(i) & lo_bit) != 0)
      ++nlo;
    if ((node_bits(i) & hi_bit) != 0)
      ++nhi;
  }
  if (nlo == 0 || nhi == 0)
    return;

  std::vector<index_t> lo(nlo), hi(nhi);
  std::vector<std::int64_t> klo(nlo), khi(nhi);
  index_t a = 0, b = 0;
  for (index_t i = 0; i < nn; ++i) {
    const bool haslo = (node_bits(i) & lo_bit) != 0;
    const bool hashi = (node_bits(i) & hi_bit) != 0;
    if (!haslo && !hashi)
      continue;
    const std::int64_t k0 = static_cast<std::int64_t>(
        (nodes(i, taxis0) - cmin(taxis0)) / dtol + real_t(0.5));
    std::int64_t k1 = 0;
    if (taxis1 >= 0)
      k1 = static_cast<std::int64_t>(
          (nodes(i, taxis1) - cmin(taxis1)) / dtol + real_t(0.5));
    const std::int64_t key = k0 * mult + k1;
    if (haslo) {
      lo[a] = i;
      klo[a] = key;
      ++a;
    }
    if (hashi) {
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

  const index_t periodicid_last =
      static_cast<index_t>(node_periodicid.size(1)) - 1;
  index_t ia = 0, ib = 0;
  while (ia < nlo && ib < nhi) {
    const std::int64_t ka = klo[olo[ia]];
    const std::int64_t kb = khi[ohi[ib]];
    if (ka == kb) {
      const index_t na = lo[olo[ia]];
      const index_t nb = hi[ohi[ib]];

      const auto cb_row = node_cellid.row(nb);
      const index_t cnt_b = cb_row(cb_row.size(0) - 1);
      for (index_t j = 0; j < cnt_b; ++j) {
        node_periodicid(na, node_fill(na)) = cb_row(j);
        node_fill(na) += 1;
      }
      node_periodicid(na, periodicid_last) = node_fill(na);

      const auto ca_row = node_cellid.row(na);
      const index_t cnt_a = ca_row(ca_row.size(0) - 1);
      for (index_t j = 0; j < cnt_a; ++j) {
        node_periodicid(nb, node_fill(nb)) = ca_row(j);
        node_fill(nb) += 1;
      }
      node_periodicid(nb, periodicid_last) = node_fill(nb);

      ++ia;
      ++ib;
    } else if (ka < kb) {
      ++ia;
    } else {
      ++ib;
    }
  }
}
