// Bindings for define_face_name. Compiled four times, once per
// manapy_domain_<float bits>_<int bits> package, with
// MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS selecting the
// precisions.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "domain_compute.hpp"

namespace {

// Same signature/argument order as the Python original; phy_faces (sorted
// in place by get_phyid), face_oldname, face_name, phyid_to_faceid and
// face_to_phyid are written in place.
void define_face_name_py(IMat phy_faces, CIVec phy_faces_name, CIMat faces,
                         CIMat node_phyfaceid, CIVec face_haloid,
                         IVec face_oldname, IVec face_name,
                         IVec phyid_to_faceid, IVec face_to_phyid) {
  define_face_name(make_view<index_t, 2>(phy_faces),
                    make_view<const index_t, 1>(phy_faces_name),
                    make_view<const index_t, 2>(faces),
                    make_view<const index_t, 2>(node_phyfaceid),
                    make_view<const index_t, 1>(face_haloid),
                    make_view<index_t, 1>(face_oldname),
                    make_view<index_t, 1>(face_name),
                    make_view<index_t, 1>(phyid_to_faceid),
                    make_view<index_t, 1>(face_to_phyid));
}

} // namespace

void register_define_face_name(nb::module_ &m) {
  m.def("define_face_name", &define_face_name_py,
        nb::arg("phy_faces").noconvert(), nb::arg("phy_faces_name"),
        nb::arg("faces"), nb::arg("node_phyfaceid"), nb::arg("face_haloid"),
        nb::arg("face_oldname").noconvert(), nb::arg("face_name").noconvert(),
        nb::arg("phyid_to_faceid").noconvert(),
        nb::arg("face_to_phyid").noconvert(),
        "Resolves every face to its physical-face id and propagates that "
        "physical face's name; a face on a halo boundary is always named "
        "10, overriding the physical name. Writes into phy_faces (sorted "
        "in place), face_oldname, face_name, phyid_to_faceid and "
        "face_to_phyid in place.");
}
