#include "domain_compute.hpp"

#include <stdexcept>

#include "common/domain_helpers.hpp"

/*
#! Here a boundary cell is a cell that is connected to a physical face.
#! It is different from a boundary cell that has a node of at least one neighbor physical face.
*/
void create_bf_cellid(ArrayView<const index_t, 2> phy_faces,
                       ArrayView<const index_t, 2> node_cellid,
                       ArrayView<const index_t, 1> phyid_to_faceid,
                       ArrayView<const index_t, 2> cell_faceid,
                       ArrayView<index_t, 1> intersect,
                       ArrayView<index_t, 2> bf_cellid) {
  const index_t nb_phy_faces = static_cast<index_t>(phy_faces.size(0));
  const index_t cell_faceid_last = cell_faceid.size(1) - 1;

  for (index_t i = 0; i < nb_phy_faces; ++i) {
    const auto phy_face = phy_faces.row(i);
    const index_t size = phy_face(phy_face.size(0) - 1);
    intersect_face_nodes(phy_face, size, node_cellid, intersect);

    const index_t cellid = intersect(0);
    const index_t faceid = phyid_to_faceid(i);
    if (cellid == -1)
      throw std::runtime_error(
          "create_bf_cellid: cellid must exist for a physical face");
    if (faceid == -1)
      throw std::runtime_error(
          "create_bf_cellid: faceid must exist for a physical face (high "
          "probability that the physical faces on the mesh are not well "
          "placed)");

    index_t face_index = -1;
    const index_t count = cell_faceid(cellid, cell_faceid_last);
    for (index_t j = 0; j < count; ++j) {
      if (cell_faceid(cellid, j) == faceid) {
        face_index = j;
        break;
      }
    }
    if (face_index == -1)
      throw std::runtime_error(
          "create_bf_cellid: faceid must exist in cell_faceid");

    bf_cellid(i, 0) = cellid;
    bf_cellid(i, 1) = face_index;
  }
}
