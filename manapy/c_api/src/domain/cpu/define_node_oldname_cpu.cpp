#include "domain_compute.hpp"

void define_node_oldname(ArrayView<const index_t, 2> phy_faces,
                          ArrayView<const index_t, 1> phy_faces_name,
                          ArrayView<index_t, 1> node_oldname) {
  const index_t nb = static_cast<index_t>(phy_faces.size(0));
  const index_t phy_last = phy_faces.size(1) - 1;

  for (index_t i = 0; i < nb; ++i) {
    const index_t name = phy_faces_name(i);
    const index_t count = phy_faces(i, phy_last);
    for (index_t j = 0; j < count; ++j) {
      const index_t nodeid = phy_faces(i, j);
      // Select the smallest name if it exists.
      if (node_oldname(nodeid) == 0 || node_oldname(nodeid) > name)
        node_oldname(nodeid) = name;
    }
  }
}
