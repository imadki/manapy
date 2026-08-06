#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

/* 
* cell_haloghostnid [[indices point to ext_ghost_info_int]]
* node_haloghostid [[indices point to ext_ghost_info_int]]

# cell_halophyid = [local_cell_id1, size, halophyid1, ..., local_cell_id2, ...]
# node_halophyid = [local_node_id1, size, halophyid1, ..., local_node_id2, ...]
*/

void create_halo_ghost_tables(ArrayView<index_t, 2> ext_ghost_info_int,
                               ArrayView<const index_t, 1> node_halophyid,
                               ArrayView<const index_t, 1> cell_halophyid,
                               ArrayView<const index_t, 2> node_haloid,
                               ArrayView<const index_t, 2> halo_halosext,
                               ArrayView<index_t, 2> cell_haloghostid,
                               ArrayView<index_t, 2> node_haloghostid) {
  const index_t cell_haloghostid_last = cell_haloghostid.size(1) - 1;
  const index_t node_haloghostid_last = node_haloghostid.size(1) - 1;

  {
    const index_t n = static_cast<index_t>(cell_halophyid.size(0));
    index_t i = 0;
    while (i + 1 < n) {
      const index_t cell_id = cell_halophyid(i);
      const index_t size = cell_halophyid(i + 1);
      index_t j = i + 2;
      const index_t end = j + size;
      while (j < end) {
        const index_t k = j - i - 2; // 0, 1, ..., size - 1
        cell_haloghostid(cell_id, k) = cell_halophyid(j);
        ++j;
      }
      cell_haloghostid(cell_id, cell_haloghostid_last) = size;
      i = end;
    }
  }

  {
    const index_t n = static_cast<index_t>(node_halophyid.size(0));
    index_t i = 0;
    while (i + 1 < n) {
      const index_t node_id = node_halophyid(i);
      const index_t size = node_halophyid(i + 1);
      index_t j = i + 2;
      const index_t end = j + size;
      while (j < end) {
        const index_t k = j - i - 2; // 0, 1, ..., size - 1
        const index_t phy_id = node_halophyid(j);
        node_haloghostid(node_id, k) = phy_id;

        // Resolve the halo cell backing this ghost, using its global id.
        const index_t cell_global_id = ext_ghost_info_int(phy_id, 2);
        ext_ghost_info_int(phy_id, 0) = search_halo_cell(
            node_haloid.row(node_id), halo_halosext, cell_global_id);
        ++j;
      }
      node_haloghostid(node_id, node_haloghostid_last) = size;
      i = end;
    }
  }
}
