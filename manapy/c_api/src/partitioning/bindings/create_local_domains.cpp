// Binding for the partitioning pipeline (src/partitioning.cpp's
// create_sub_domains), ported from the c_api's py_create_local_domains.
//
// The original's bookkeeping is gone: no PyArray_FROM_OTF per argument, no
// free_tables() lambda, no Py_XDECREF on each of eight error paths, no
// try/catch translating std::exception into PyErr_SetString. nanobind converts
// the arguments, unique_ptr owns the partition descriptors, and any throw --
// from here or from deep inside the pipeline -- becomes the matching Python
// exception on its own.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "partitioning.hpp"

#include <memory>
#include <stdexcept>

namespace {

nb::list create_local_domains_py(CIVec part_vert, CIMat node_cellid,
                                 CIMat node_phyid, CIMat cells,
                                 CI8Vec cells_type, CFMat nodes,
                                 CIMat phy_faces, CIVec phy_faces_name,
                                 index_t nb_parts, index_t dim) {
  if (nb_parts < 2)
    throw std::invalid_argument("nb_parts must be ≥ 2");
  if (dim != 2 and dim != 3)
    throw std::invalid_argument("dim must be 2 or 3");

  auto ld = std::make_unique<LocalDomainStruct[]>(nb_parts);

  // Argument order follows create_sub_domains, which differs from the Python
  // signature below (nodes and cells are swapped) -- as it did in the c_api.
  return create_sub_domains(
      ld.get(), make_view<const index_t, 1>(part_vert),
      make_view<const index_t, 2>(node_cellid),
      make_view<const real_t, 2>(nodes), make_view<const index_t, 2>(cells),
      make_view<const std::int8_t, 1>(cells_type),
      make_view<const index_t, 2>(phy_faces),
      make_view<const index_t, 1>(phy_faces_name),
      make_view<const index_t, 2>(node_phyid), nb_parts, dim);
}

/* ------------------------------------------------------------------------- */
/*  Docstring for py_create_local_domains                                    */
/* ------------------------------------------------------------------------- */
// Note: the c_api's version of this docstring described a 22-item tuple. That
// was stale -- LocalDomainStruct::create_tuple has always built 27 (18 arrays
// + 9 scalars). The list below is the real thing.
const char create_local_domains_doc[] = R"doc(
create_local_domains(part_vert,
                     node_cellid,
                     node_phyid,
                     cells,
                     cells_type,
                     nodes,
                     phy_faces,
                     phy_faces_name,
                     nb_parts,
                     dim) -> list[tuple]

Partition an unstructured mesh into *nb_parts* sub-domains and build all
per-partition connectivity / halo tables needed by the solver.

Parameters
----------
part_vert : numpy.ndarray[idx_t]               (n_vertices,)
    Partition id of every vertex **before** repartitioning.
node_cellid : numpy.ndarray[idx_t]             (n_vertices,)
    Global cell id that first owns each vertex.
node_phyid : numpy.ndarray[idx_t]              (n_vertices,)
    Physical (boundary-condition) id attached to each vertex.
cells : numpy.ndarray[idx_t]                   (n_cells, max_nodes_per_cell)
    Node connectivity of each cell (global node indices).
cells_type : numpy.ndarray[int8]               (n_cells,)
    Element-type code per cell (e.g. 5 = tetra, 9 = hex, ...).
nodes : numpy.ndarray[fdx_t]         (n_vertices, 3)
    Cartesian coordinates of every node. Must have exactly 3 columns.
phy_faces : numpy.ndarray[idx_t]               (n_phy_faces, max_nodes_per_face)
    Connectivity of boundary faces (global node indices).
phy_faces_name : numpy.ndarray[idx_t]          (n_phy_faces,)
    Physical-name / BC id of each boundary face.
nb_parts : idx_t
    Number of sub-domains to create (must be >= 2).
dim : idx_t
    Dimension of the mesh (2 or 3).

Returns
-------
list[tuple]
    A Python list of length `nb_parts`.
    The `p`-th element (`parts[p]`) is a 27-item tuple containing
    every array that belongs to partition `p`.

    0. nodes               - fdx_t (n_nodes_p, 3)
    1. cells               - idx_t (n_cells_p, max_cell_nodeid + 1)
    2. cells_type          - int8  (n_cells_p,)
    3. phy_faces           - idx_t (n_phy_faces_p, max_phy_face_nodeid + 1)
    4. phy_faces_name      - idx_t (n_phy_faces_p,)
    5. cell_loctoglob      - idx_t (n_cells_p,)
    6. node_loctoglob      - idx_t (n_nodes_p,)
    7. node_oldname        - idx_t (n_nodes_p,)
    8. halo_neighsub       - idx_t (2, n_neigh_parts_p)
    9. node_halos          - idx_t (2 * n_ext_halo_nodes_p,)
   10. halo_halosext       - idx_t (n_halos_p, max_halo_cell_nodeid + 2)
   11. halo_halosint       - idx_t (n_halos_int_p,)
   12. halo_centvol        - fdx_t (n_halos_p, 4)
   13. phyid_neighbor      - idx_t [[Neighbor partition ID, nb_send, nb_recv] ...]
   14. phyid_recv          - idx_t [PhyFaceGlobalId, ...]
   15. phyid_send          - idx_t [PhyFaceLocalId], ...
   16. node_halophyid      - idx_t [NodeLocalId1, Size1, Size1 indices into phyid_recv..., ...]
   17. cell_halophyid      - idx_t [CellLocalId1, Size1, Size1 indices into phyid_recv..., ...]
   18. max_cell_nodeid     - int
   19. max_cell_faceid     - int
   20. max_face_nodeid     - int
   21. max_node_haloid     - int
   22. max_cell_halonid    - int
   23. max_node_phyid      - int
   24. max_node_halophyid  - int
   25. max_cell_phyid      - int
   26. max_cell_halophyid  - int

Notes
-----
* All arrays are new NumPy objects; none of the inputs are modified.
* Array dtypes and shapes are guaranteed as shown; callers may rely on them.
* max_halo_cell_nodeid and max_phy_face_nodeid are internal sizing values and
  are deliberately not part of the tuple; they are implied by the widths of
  halo_halosext and phy_faces.
)doc";

} // namespace

void register_create_local_domains(nb::module_ &m) {
  m.def("create_local_domains", &create_local_domains_py, nb::arg("part_vert"),
        nb::arg("node_cellid"), nb::arg("node_phyid"), nb::arg("cells"),
        nb::arg("cells_type"), nb::arg("nodes"), nb::arg("phy_faces"),
        nb::arg("phy_faces_name"), nb::arg("nb_parts"), nb::arg("dim"),
        create_local_domains_doc);
}
