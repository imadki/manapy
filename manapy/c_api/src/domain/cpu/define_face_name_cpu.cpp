#include "domain_compute.hpp"

#include "common/domain_helpers.hpp"

void define_face_name(ArrayView<index_t, 2> phy_faces,
                       ArrayView<const index_t, 1> phy_faces_name,
                       ArrayView<const index_t, 2> faces,
                       ArrayView<const index_t, 2> node_phyfaceid,
                       ArrayView<const index_t, 1> face_haloid,
                       ArrayView<index_t, 1> face_oldname,
                       ArrayView<index_t, 1> face_name,
                       ArrayView<index_t, 1> phyid_to_faceid,
                       ArrayView<index_t, 1> face_to_phyid) {
  const index_t nb_faces = static_cast<index_t>(faces.size(0));
  const bool has_haloid = face_haloid.size(0) != 0;

  for (index_t i = 0; i < nb_faces; ++i) {
    const index_t phyid = get_phyid(phy_faces, faces.row(i), node_phyfaceid);
    index_t name = 0;
    if (phyid != -1) {
      phyid_to_faceid(phyid) = i;
      name = phy_faces_name(phyid);
    }

    face_to_phyid(i) = phyid;
    face_oldname(i) = name;
    face_name(i) = name;
    if (has_haloid && face_haloid(i) != -1)
      face_name(i) = 10;
  }
}
