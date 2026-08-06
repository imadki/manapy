// The METIS calls behind the three make_n_part_* entry points, ported from the
// c_api's src/py_manapy_part.cpp (where they were file-static helpers sitting
// with the bindings; the Python glue is now in
// bindings/metis_partitioning.cpp).
//
// Ownership: the partition vector METIS fills is an OwnedArray, so it is freed
// automatically if METIS reports an error, and handed to NumPy without a copy
// on success -- what the original did with malloc plus
// PyArray_SimpleNewFromData + PyArray_ENABLEFLAGS(NPY_ARRAY_OWNDATA), minus the
// manual free on every error path. The CSR scratch is std::vector for the same
// reason.

#include "metis_partitioning.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * @brief Converts a nested integer NumPy array representation of a graph into a Compressed Sparse Row (CSR) format.
 *
 * @details This function takes a 2D NumPy array where each row represents a vertex, and the columns contain
 * the neighbor indices, and the last column explicitly holds the degree (size) of that vertex's neighborhood.
 * It allocates and populates the `xadj` (row pointer) and `adjncy` (column indices) arrays required by METIS.
 *
 * @param graph A 2D NumPy array of size (nb_vertices, max_cols).
 * @param xadj Output pointer for the xadj array of size `nvtxs + 1` allocated by malloc.
 * @param adjncy Output pointer for the adjncy array of size `total_deg` allocated by malloc.
 * @param nvtxs Output pointer representing the total number of vertices in the mesh.
 * @param nb_nodes Output pointer storing the maximum node id encountered + 1.
 * @param total_deg Output pointer representing the total number of edges (sum of degrees).
 * @return int 0 on success, or -1 on error (e.g., MemoryError or ValueError).
 */
void dense_to_csr(ArrayView<const index_t, 2> graph, std::vector<index_t> &xadj,
                  std::vector<index_t> &adjncy, index_t &nvtxs,
                  index_t &nb_nodes, index_t &total_deg) {
    const index_t nb_vertices = static_cast<index_t>(graph.size(0));
    const index_t max_cols = static_cast<index_t>(graph.size(1));
    if (max_cols < 1) {
        throw std::invalid_argument("graph must have at least one column (size field)");
    }

    // Get total_deg and nvtxs
    index_t deg_sum = 0;
    for (index_t vertex = 0; vertex < nb_vertices; vertex++) {
        const index_t size = graph(vertex, max_cols - 1);
        deg_sum += size;
    }
    total_deg = deg_sum;
    nvtxs = nb_vertices;




    //Get xadj and adjncy
    xadj.resize(nb_vertices + 1);
    adjncy.resize(deg_sum);

    nb_nodes = 0;
    index_t counter = 0;
    xadj[0] = 0;
    for (index_t i = 0; i < nb_vertices; i++) {
        const index_t deg = graph(i, max_cols - 1);
        for (index_t j = 0; j < deg; j++) {
            const index_t nb = graph(i, j);
            adjncy[counter++] = nb;
            if (nb > nb_nodes) {
                // in the case of metis nodal or dual neighbor represent the node
                nb_nodes = nb;
            }
        }
        xadj[i + 1] = counter;
    }
    nb_nodes++;

    MANAPY_PRINT_DEBUG("Nb_cells = %ld\n", (long) nb_vertices);
    MANAPY_PRINT_DEBUG("total_deg size = %ld\n", (long) deg_sum);
    MANAPY_PRINT_DEBUG("Nb_nodes = %ld\n", (long) nb_nodes);
}

// METIS reports failure through a status code; turn it into the RuntimeError
// the c_api raised with PyErr_Format.
void check_metis(const index_t ret, const char *what) {
    if (ret != METIS_OK)
        throw std::runtime_error(std::string(what) + " failed (status=" +
                                 std::to_string(ret) + ")");
}

} // namespace

/**
 * @brief Partitions a graph into `k` parts using the METIS standard k-way partitioning algorithm.
 *
 * @details Translates the input dense structure into CSR format and runs `METIS_PartGraphKway`.
 * The resulting partitioning scheme is returned as a 1D array allocated internally.
 *
 * @param graph 2D PyArrayObject holding adjacency and degree data.
 * @param nb_part Number of partitions to divide the graph into.
 * @param part_vert Output pointer to a newly allocated array of size `nvtxs` containing partition IDs per vertex.
 * @return int 0 on success, -1 on failure.
 */
OwnedArray<index_t, 1> make_n_part_graph_k_way(ArrayView<const index_t, 2> graph,
                                               index_t nb_part) {
    std::vector<index_t> xadj;
    std::vector<index_t> adjncy;
    index_t nvtxs;
    index_t deg_sum;
    index_t ret;
    index_t nb_node;


    dense_to_csr(graph, xadj, adjncy, nvtxs, nb_node, deg_sum);


    OwnedArray<index_t, 1> part_idx(make_dims(nvtxs));

    index_t ncon = 1;
    index_t edgecut = 0;

    MANAPY_PRINT_DEBUG("METIS_PartGraphKway");
    ret = METIS_PartGraphKway(&nvtxs, &ncon,
                                 xadj.data(), adjncy.data(),
                                 nullptr, nullptr, nullptr,
                                 &nb_part,
                                 nullptr, nullptr,
                                 nullptr,
                                 &edgecut, part_idx.data());

    check_metis(ret, "METIS_PartGraphKway");

    return part_idx;
}

/**
 * @brief Partitions a mesh using its dual graph formulation via METIS.
 *
 * @details Two elements are connected in the dual graph if they share at least `n_common` nodes.
 *
 * @param cells 2D PyArrayObject defining the elements and their constituent nodes.
 * @param nb_part Number of partitions to create.
 * @param n_common Minimum number of shared nodes required to form an edge in the dual graph.
 * @param part_vert Output pointer to a dynamically allocated array for element partition assignments.
 * @return int 0 on success, -1 on array allocation failure or METIS runtime error.
 */
OwnedArray<index_t, 1> make_n_part_mesh_dual(ArrayView<const index_t, 2> cells,
                                             index_t nb_part, index_t n_common) {
    std::vector<index_t> eptr;
    std::vector<index_t> eind;
    index_t ne;
    index_t deg_sum;
    index_t objval;
    index_t ret;
    index_t nb_nodes;
    index_t options[METIS_NOPTIONS];

    dense_to_csr(cells, eptr, eind, ne, nb_nodes, deg_sum);

    OwnedArray<index_t, 1> part_idx(make_dims(ne));
    std::vector<index_t> npart(nb_nodes);



    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // C-style indexing (0-based)
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // or METIS_OBJTYPE_VOL
    // options[METIS_OPTION_CONTIG] = 1; // enforce contiguous partitions

    MANAPY_PRINT_DEBUG("METIS_PartMeshDual");
    ret = METIS_PartMeshDual(
        &ne, // number of elements
        &nb_nodes, // number of nodes
        eptr.data(), // element pointer array (CSR-style)
        eind.data(), // element connectivity (node indices)
        nullptr, // vwgt optional: weights for elements
        nullptr, // vsize optional: sizes for elements
        &n_common, // number of common nodes to define adjacency
        &nb_part, // number of partitions
        nullptr,  // tpwgts optional: target partition weights
        options, // array of options
        &objval, // output: edgecut or communication volume
        part_idx.data(), // output: element partition assignment
        npart.data() // output: node partition assignment
        );

    check_metis(ret, "METIS_PartMeshDual");

    return part_idx;
}

/**
 * @brief Partitions a mesh using a nodal graph formulation via METIS.
 *
 * @details Creates partitions focusing on balancing nodes among the partitions and minimizing node connection cuts.
 *
 * @param cells 2D PyArrayObject holding the element/node connectivity and degrees.
 * @param nb_part Number of partitions.
 * @param part_vert Output pointer for the resulting element partition IDs.
 * @return int 0 on success, -1 on array allocation failure or METIS runtime error.
 */
OwnedArray<index_t, 1> make_n_part_mesh_nodal(ArrayView<const index_t, 2> cells,
                                              index_t nb_part) {
    std::vector<index_t> eptr;
    std::vector<index_t> eind;
    index_t ne;
    index_t deg_sum;
    index_t objval;
    index_t ret;
    index_t nb_nodes;
    index_t options[METIS_NOPTIONS];


    dense_to_csr(cells, eptr, eind, ne, nb_nodes, deg_sum);

    OwnedArray<index_t, 1> part_idx(make_dims(ne));
    std::vector<index_t> npart(nb_nodes);


    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // C-style indexing (0-based)
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // or METIS_OBJTYPE_VOL
    // options[METIS_OPTION_CONTIG] = 1; // enforce contiguous partitions

    MANAPY_PRINT_DEBUG("METIS_PartMeshNodal\n");
    ret = METIS_PartMeshNodal(
        &ne, // number of elements
        &nb_nodes, // number of nodes
        eptr.data(), // element pointer array (CSR-style)
        eind.data(), // element connectivity (node indices)
        nullptr, // vwgt optional: weights for elements
        nullptr, // vsize optional: sizes for elements
        &nb_part, // number of partitions
        nullptr,  // tpwgts optional: target partition weights
        options, // array of options
        &objval, // output: edgecut or communication volume
        part_idx.data(), // output: element partition assignment
        npart.data() // output: node partition assignment
        );

    check_metis(ret, "METIS_PartMeshNodal");

    return part_idx;
}
