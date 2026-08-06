#pragma once

// The three METIS wrappers, implemented in src/metis_partitioning.cpp. In the
// c_api these were file-static helpers inside py_manapy_part.cpp, next to the
// bindings that expose them; here the implementation sits with the rest of the
// module's source and bindings/metis_partitioning.cpp holds only the Python
// glue. Full parameter docs are on the definitions.

#include "manapy_part.hpp"

// METIS_PartGraphKway over a dense adjacency matrix whose last column holds
// each vertex's degree. Returns one partition id per vertex.
OwnedArray<index_t, 1> make_n_part_graph_k_way(ArrayView<const index_t, 2> graph,
                                               index_t nb_part);

// METIS_PartMeshDual: two elements are adjacent when they share at least
// n_common nodes. Returns one partition id per element.
OwnedArray<index_t, 1> make_n_part_mesh_dual(ArrayView<const index_t, 2> cells,
                                             index_t nb_part, index_t n_common);

// METIS_PartMeshNodal. Returns one partition id per element.
OwnedArray<index_t, 1> make_n_part_mesh_nodal(ArrayView<const index_t, 2> cells,
                                              index_t nb_part);
