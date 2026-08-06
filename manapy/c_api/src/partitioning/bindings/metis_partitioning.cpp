// Python glue for the three METIS wrappers. The METIS calls themselves are in
// src/metis_partitioning.cpp; this file only validates arguments, converts
// arrays and carries the docstrings -- the binding half of what the c_api kept
// together in py_manapy_part.cpp.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"
#include "metis_partitioning.hpp"

#include <stdexcept>

namespace {

//---------------------------------------------------------------------------------------
//Python Binding functions
//---------------------------------------------------------------------------------------

nb::object make_n_part_graph_k_way_py(CIMat graph, index_t nb_parts) {
    if (nb_parts < 2) {
        throw std::invalid_argument("nb_parts must be ≥ 2");
    }

    MANAPY_PRINT_DEBUG_TIME_START();
    auto part_vert = make_n_part_graph_k_way(make_view<const index_t, 2>(graph), nb_parts);
    MANAPY_PRINT_DEBUG_TIME("make_n_part_graph_k_way");

    return nb::cast(part_vert.release());
}

nb::object make_n_part_mesh_dual_py(CIMat cells, index_t nb_parts, index_t n_common) {
    if (nb_parts < 2) {
        throw std::invalid_argument("nb_parts must be ≥ 2");
    }

    MANAPY_PRINT_DEBUG_TIME_START();
    auto part_vert = make_n_part_mesh_dual(make_view<const index_t, 2>(cells), nb_parts, n_common);
    MANAPY_PRINT_DEBUG_TIME("make_n_part_mesh_dual");

    return nb::cast(part_vert.release());
}

nb::object make_n_part_mesh_nodal_py(CIMat cells, index_t nb_parts) {
    if (nb_parts < 2) {
        throw std::invalid_argument("nb_parts must be ≥ 2");
    }

    MANAPY_PRINT_DEBUG_TIME_START();
    auto part_vert = make_n_part_mesh_nodal(make_view<const index_t, 2>(cells), nb_parts);
    MANAPY_PRINT_DEBUG_TIME("make_n_part_mesh_nodal");

    return nb::cast(part_vert.release());
}

/* -------- module definition --------------------------------------- */
// ----------------- Method Table -----------------------

const char doc_make_n_part_graph_k_way[] = R"doc(
make_n_part_graph_k_way(graph, nb_part) -> numpy.ndarray

Partition a graph into `nb_part` parts using METIS_PartGraphKway.

Parameters
----------
graph : numpy.ndarray[idx_t]
    A 2D adjacency matrix of size `(n_vertices, max_cell_neighbors)`.
    The last element of each row contains the node degree (number of neighbors).
nb_part : int32
    Number of partitions to create (must be >= 2).

Returns
-------
numpy.ndarray[idx_t]
    A 1D array of shape `(n_vertices,)` containing the partition ID for each vertex.
)doc";

const char doc_make_n_part_mesh_dual[] = R"doc(
make_n_part_mesh_dual(cells, nb_parts, n_common) -> numpy.ndarray

Partition a mesh into `nb_parts` parts using the dual graph formulation (METIS_PartMeshDual).
Two elements are considered adjacent if they share at least `n_common` nodes.

Parameters
----------
cells : numpy.ndarray[idx_t]
    A 2D array of element connectivity of size `(n_elements, max_nodes_per_element)`.
    The last element in each row contains the element degree (number of nodes).
nb_parts : int32
    Number of partitions to create (must be >= 2).
n_common : int32
    Number of common nodes required to define adjacency between two elements.

Returns
-------
numpy.ndarray[idx_t]
    A 1D array of shape `(n_elements,)` containing the partition ID for each element.
)doc";

const char doc_make_n_part_mesh_nodal[] = R"doc(
make_n_part_mesh_nodal(cells, nb_parts) -> numpy.ndarray

Partition a mesh into `nb_parts` parts using the nodal graph formulation (METIS_PartMeshNodal).

Parameters
----------
cells : numpy.ndarray[idx_t]
    A 2D array of element connectivity of size `(n_elements, max_nodes_per_element)`.
    The last element in each row contains the element degree (number of nodes).
nb_parts : int32
    Number of partitions to create (must be >= 2).

Returns
-------
numpy.ndarray[idx_t]
    A 1D array of shape `(n_elements,)` containing the partition ID for each element.
)doc";

} // namespace

void register_metis_partitioning(nb::module_ &m) {
  m.def("make_n_part_graph_k_way", &make_n_part_graph_k_way_py,
        nb::arg("graph"), nb::arg("nb_part"), doc_make_n_part_graph_k_way);
  m.def("make_n_part_mesh_dual", &make_n_part_mesh_dual_py, nb::arg("cells"),
        nb::arg("nb_parts"), nb::arg("n_common"), doc_make_n_part_mesh_dual);
  m.def("make_n_part_mesh_nodal", &make_n_part_mesh_nodal_py, nb::arg("cells"),
        nb::arg("nb_parts"), doc_make_n_part_mesh_nodal);
}
