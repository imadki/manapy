#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#include <numpy/arrayobject.h>
#include "manapy_part.h"

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
static int dense_to_csr(PyArrayObject *graph, idx_t **xadj, idx_t **adjncy, idx_t *nvtxs, idx_t *nb_nodes, idx_t *total_deg) {
    const npy_intp *dims = PyArray_DIMS(graph);
    const idx_t nb_vertices = (idx_t) dims[0];
    const idx_t max_cols = (idx_t)dims[1];
    if (max_cols < 1) {
        PyErr_SetString(PyExc_ValueError, "graph must have at least one column (size field)");
        return -1;
    }

    // Get total_deg and nvtxs
    idx_t deg_sum = 0;
    for (idx_t vertex = 0; vertex < nb_vertices; vertex++) {
        const idx_t size = *(idx_t *)PyArray_GETPTR2(graph, vertex, max_cols - 1);
        deg_sum += size;
    }
    *total_deg = deg_sum;
    *nvtxs = nb_vertices;




    //Get xadj and adjncy
    *xadj = (idx_t *)malloc(sizeof(idx_t) * (nb_vertices + 1));
    *adjncy = (idx_t *)malloc(sizeof(idx_t) * (deg_sum));
    if (*xadj == nullptr || *adjncy == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }

    *nb_nodes = 0;
    idx_t counter = 0;
    (*xadj)[0] = 0;
    for (idx_t i = 0; i < nb_vertices; i++) {
        const idx_t deg = *(idx_t *)PyArray_GETPTR2(graph, i, max_cols - 1);
        for (idx_t j = 0; j < deg; j++) {
            const idx_t nb = *(idx_t *)PyArray_GETPTR2(graph, i, j);
            (*adjncy)[counter++] = nb;
            if (nb > *nb_nodes) {
                // in the case of metis nodal or dual neighbor represent the node
                *nb_nodes = nb;
            }
        }
        (*xadj)[i + 1] = counter;
    }
    (*nb_nodes)++;

    DEBUG_PRINT_INSTANT("Nb_vertices = %ld\n", (long) nb_vertices);
    DEBUG_PRINT_INSTANT("total_deg size = %d\n", (long) deg_sum);
    DEBUG_PRINT_INSTANT("Nb_nodes = %d\n", (long) (*nb_nodes));
    return 0;

}

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
static int make_n_part_graph_k_way(PyArrayObject *graph, idx_t nb_part, idx_t **part_vert) {
    idx_t *xadj;
    idx_t *adjncy;
    idx_t nvtxs;
    idx_t deg_sum;
    idx_t ret;
    idx_t nb_node;


    ret = dense_to_csr(graph, &xadj, &adjncy, &nvtxs, &nb_node, &deg_sum);
    if (ret < 0)
        return -1;


    idx_t *part_idx = (idx_t *)malloc(sizeof(idx_t) * nvtxs);
    if (part_idx == nullptr) {
        free(xadj);
        free(adjncy);
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }

    idx_t ncon = 1;
    idx_t edgecut = 0;

    DEBUG_PRINT_INSTANT("METIS_PartGraphKway");
    ret = METIS_PartGraphKway(&nvtxs, &ncon,
                                 xadj, adjncy,
                                 nullptr, nullptr, nullptr,
                                 &nb_part,
                                 nullptr, nullptr,
                                 nullptr,
                                 &edgecut, part_idx);

    free(xadj);
    free(adjncy);

    if (ret != METIS_OK) {
        free(part_idx);
        PyErr_Format(PyExc_RuntimeError, "METIS_PartGraphKway failed (status=%d)", ret);
        return -1;
    }


    *part_vert = part_idx;
    return 0;
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
static int make_n_part_mesh_dual(PyArrayObject *cells, idx_t nb_part, idx_t n_common, idx_t **part_vert) {
    idx_t *eptr;
    idx_t *eind;
    idx_t ne;
    idx_t deg_sum;
    idx_t objval;
    idx_t ret;
    idx_t nb_nodes;
    idx_t options[METIS_NOPTIONS];

    if (dense_to_csr(cells, &eptr, &eind, &ne, &nb_nodes, &deg_sum) < 0)
        return -1;

    idx_t *part_idx = (idx_t *)malloc(sizeof(idx_t) * ne);
    idx_t *npart = (idx_t *)malloc(sizeof(idx_t) * nb_nodes);
    if (part_idx == nullptr || npart == nullptr) {
        free(eptr);
        free(eind);
        free(part_idx);
        free(npart);
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }



    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // C-style indexing (0-based)
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // or METIS_OBJTYPE_VOL
    // options[METIS_OPTION_CONTIG] = 1; // enforce contiguous partitions

    DEBUG_PRINT_INSTANT("METIS_PartMeshDual");
    ret = METIS_PartMeshDual(
        &ne, // number of elements
        &nb_nodes, // number of nodes
        eptr, // element pointer array (CSR-style)
        eind, // element connectivity (node indices)
        nullptr, // vwgt optional: weights for elements
        nullptr, // vsize optional: sizes for elements
        &n_common, // number of common nodes to define adjacency
        &nb_part, // number of partitions
        nullptr,  // tpwgts optional: target partition weights
        options, // array of options
        &objval, // output: edgecut or communication volume
        part_idx, // output: element partition assignment
        npart // output: node partition assignment
        );

    free(eptr);
    free(eind);
    free(npart);

    if (ret != METIS_OK) {
        free(part_idx);
        PyErr_Format(PyExc_RuntimeError, "METIS_PartGraphKway failed (status=%d)", ret);
        return -1;
    }

    *part_vert = part_idx;
    return 0;
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
static int make_n_part_mesh_nodal(PyArrayObject *cells, idx_t nb_part, idx_t **part_vert) {
    idx_t *eptr;
    idx_t *eind;
    idx_t ne;
    idx_t deg_sum;
    idx_t objval;
    idx_t ret;
    idx_t nb_nodes;
    idx_t options[METIS_NOPTIONS];


    if (dense_to_csr(cells, &eptr, &eind, &ne, &nb_nodes, &deg_sum) < 0)
        return -1;

    idx_t *part_idx = (idx_t *)malloc(sizeof(idx_t) * ne);
    idx_t *npart = (idx_t *)malloc(sizeof(idx_t) * nb_nodes);
    if (part_idx == nullptr || npart == nullptr) {
        free(eptr);
        free(eind);
        free(part_idx);
        free(npart);
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }


    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // C-style indexing (0-based)
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // or METIS_OBJTYPE_VOL
    // options[METIS_OPTION_CONTIG] = 1; // enforce contiguous partitions

    DEBUG_PRINT_INSTANT("METIS_PartMeshNodal\n");
    ret = METIS_PartMeshNodal(
        &ne, // number of elements
        &nb_nodes, // number of nodes
        eptr, // element pointer array (CSR-style)
        eind, // element connectivity (node indices)
        nullptr, // vwgt optional: weights for elements
        nullptr, // vsize optional: sizes for elements
        &nb_part, // number of partitions
        nullptr,  // tpwgts optional: target partition weights
        options, // array of options
        &objval, // output: edgecut or communication volume
        part_idx, // output: element partition assignment
        npart // output: node partition assignment
        );

    free(eptr);
    free(eind);
    free(npart);

    if (ret != METIS_OK) {
        free(part_idx);
        PyErr_Format(PyExc_RuntimeError, "METIS_PartGraphKway failed (status=%d)", ret);
        return -1;
    }

    *part_vert = part_idx;
    return 0;
}

//---------------------------------------------------------------------------------------
//Python Binding functions
//---------------------------------------------------------------------------------------

/**
 * @brief Python binding for `make_n_part_graph_k_way`.
 * @details Exposed Python interface taking `graph` (numpy matrix) and `nb_parts` (int).
 * @param self The module object.
 * @param args Python tuple `(graph, nb_parts)`.
 * @return PyObject* A 1D integer NumPy array of partition IDs.
 */
static PyObject *py_make_n_part_graph_k_way(PyObject *self, PyObject *args) {
    PyObject *graph_obj = nullptr;
    int nb_parts = 0;

    if (!PyArg_ParseTuple(args, "Oi", &graph_obj, &nb_parts))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *graph = (PyArrayObject *)PyArray_FROM_OTF(graph_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!graph)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;

    DEBUG_TIME_IT("");
    ret = make_n_part_graph_k_way(graph, nb_parts, &part_vert);
    DEBUG_TIME_IT("make_n_part_graph_k_way");
    if (ret == -1) {
        Py_DECREF(graph);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(graph)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, part_vert);
    if (!part_array) {
        Py_DECREF(graph);
        return nullptr;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    Py_DECREF(graph);

    PyObject *ret_data = Py_BuildValue("O", part_array);
    if (!ret_data)
        Py_DECREF(part_array);
    return ret_data;
}

/**
 * @brief Python binding for `make_n_part_mesh_dual`.
 * @details Exposed Python interface taking `cells` (numpy matrix), `nb_parts`, and `n_common`.
 * @param self The module object.
 * @param args Python tuple `(cells, nb_parts, n_common)`.
 * @return PyObject* A 1D integer NumPy array of partition IDs.
 */
static PyObject *py_make_n_part_mesh_dual(PyObject *self, PyObject *args) {
    PyObject *cells_obj = nullptr;
    int nb_parts = 0;
    int n_common = 0;

    if (!PyArg_ParseTuple(args, "Oii", &cells_obj, &nb_parts, &n_common))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!cells)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;


    DEBUG_TIME_IT("");
    ret = make_n_part_mesh_dual(cells, nb_parts, n_common, &part_vert);
    DEBUG_TIME_IT("make_n_part_mesh_dual");
    if (ret == -1) {
        Py_DECREF(cells);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(cells)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, part_vert);
    if (!part_array) {
        Py_DECREF(cells);
        return nullptr;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    Py_DECREF(cells);

    PyObject *ret_data = Py_BuildValue("O", part_array);
    if (!ret_data)
        Py_DECREF(part_array);
    return ret_data;
}

/**
 * @brief Python binding for `make_n_part_mesh_nodal`.
 * @details Exposed Python interface taking `cells` (numpy matrix) and `nb_parts`.
 * @param self The module object.
 * @param args Python tuple `(cells, nb_parts)`.
 * @return PyObject* A 1D integer NumPy array of partition IDs.
 */
static PyObject *py_make_n_part_mesh_nodal(PyObject *self, PyObject *args) {
    PyObject *cells_obj = nullptr;
    int nb_parts = 0;

    if (!PyArg_ParseTuple(args, "Oi", &cells_obj, &nb_parts))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!cells)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;


    DEBUG_TIME_IT("");
    ret = make_n_part_mesh_nodal(cells, nb_parts, &part_vert);
    DEBUG_TIME_IT("make_n_part_mesh_nodal");
    if (ret == -1) {
        Py_DECREF(cells);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(cells)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, part_vert);
    if (!part_array) {
        Py_DECREF(cells);
        free(part_array);
        return nullptr;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    Py_DECREF(cells);

    PyObject *ret_data = Py_BuildValue("O", part_array);
    if (!ret_data)
        Py_DECREF(part_array);
    return ret_data;
}

/**
 * @brief Python binding to partition an unstructured mesh and build sub-domain representations.
 * @details Gathers global inputs such as node-to-cell maps, node coordinate arrays, connectivity tables,
 * and distributes them into domain-local tables arrays. Uses PyArrayObject to efficiently interface with Python.
 * @param self The module object.
 * @param args Python tuple containing elements matching `create_local_domains` documentation.
 * @return PyObject* A Python list containing a sequence of tuples storing arrays for each sub-domain.
 */
static PyObject *py_create_local_domains(PyObject *self, PyObject *args) {
    PyObject *part_vert_obj = nullptr;
    PyObject *node_cellid_obj = nullptr;
    PyObject *node_phyid_obj = nullptr;
    PyObject *cells_obj = nullptr;
    PyObject *cells_type_obj = nullptr;
    PyObject *nodes_obj = nullptr;
    PyObject *phy_faces_obj = nullptr;
    PyObject *phy_faces_name_obj = nullptr;
    idx_t nb_parts = 0;
    idx_t dim;

    if (!PyArg_ParseTuple(args, "OOOOOOOOii", &part_vert_obj, &node_cellid_obj, &node_phyid_obj, &cells_obj, &cells_type_obj, &nodes_obj, &phy_faces_obj, &phy_faces_name_obj, &nb_parts, &dim))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }
    if (dim != 2 and dim != 3) {
        PyErr_SetString(PyExc_ValueError, "dim must be 2 or 3");
        return nullptr;
    }

    /*
    *Use NPY_ARRAY_IN_ARRAY when you:
        Only read the data.
        Need it aligned and C-contiguous.
        Want NumPy to copy if necessary and handle the details for you.
     */

    PyArrayObject *part_vert = (PyArrayObject *)PyArray_FROM_OTF(part_vert_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_cellid = (PyArrayObject *)PyArray_FROM_OTF(node_cellid_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_phyid = (PyArrayObject *)PyArray_FROM_OTF(node_phyid_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells_type = (PyArrayObject *)PyArray_FROM_OTF(cells_type_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nodes = (PyArrayObject *)PyArray_FROM_OTF(nodes_obj, NPY_FLOAT_TYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces_name = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_name_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    auto *local_domains = new(std::nothrow) LocalDomainStruct[nb_parts];


    const auto free_tables = [&]() {
        Py_XDECREF(part_vert); part_vert = nullptr;
        Py_XDECREF(node_cellid); node_cellid = nullptr;
        Py_XDECREF(node_phyid); node_phyid = nullptr;
        Py_XDECREF(cells); cells = nullptr;
        Py_XDECREF(cells_type); cells_type = nullptr;
        Py_XDECREF(nodes); nodes = nullptr;
        Py_XDECREF(phy_faces); phy_faces = nullptr;
        Py_XDECREF(phy_faces_name); phy_faces_name = nullptr;
        delete []local_domains; local_domains = nullptr;
    };

    if (!part_vert || !node_cellid || !node_phyid || !cells || !cells_type || !nodes || !phy_faces || !phy_faces_name || !local_domains) {
        free_tables();
        return nullptr;
    }



    try {
        PyArray<int32_t, 1> py_part_vert(part_vert);
        PyArray<int32_t, 2> py_node_cellid(node_cellid);
        PyArray<int32_t, 2> py_node_phyid(node_phyid);
        PyArray<int32_t, 2> py_cells(cells);
        PyArray<int8_t, 1> py_cells_type(cells_type);
        PyArray<fdx_t, 2> py_nodes(nodes);
        PyArray<int32_t, 2> py_phy_faces(phy_faces);
        PyArray<int32_t, 1> py_phy_faces_name(phy_faces_name);

        PyObject *py_list_result_tmp = create_sub_domains(local_domains, &py_part_vert, &py_node_cellid, &py_nodes, &py_cells, &py_cells_type, &py_phy_faces, &py_phy_faces_name, &py_node_phyid, nb_parts, dim);


        // Free resources and return
        free_tables();
        return py_list_result_tmp;
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        // Free resources
        free_tables();
        return nullptr;
    }
}

/**
 * @brief General (2D/3D) geometric utility binding to calculate cell centroids and size (area or volume).
 * @details Invokes the internal C++ template methods based on the `dim` parameter.
 * @param self The module object.
 * @param args Python tuple containing `cells`, `nodes`, `cell_size_array`, `cell_center_array`.
 * @param dim Spatial dimension count (2 or 3) indicating flat area vs solid volume compute.
 * @return PyObject* Always returns Py_NAN, but executes in-place operations on the geometry buffers.
 */
static PyObject *py_compute_cell_center_area_volume_general(PyObject *self, PyObject *args, const int32_t dim) {
    PyObject *cells_obj = nullptr;
    PyObject *nodes_obj = nullptr;
    PyObject *cell_area_obj = nullptr;
    PyObject *cell_center_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOO", &cells_obj, &nodes_obj, &cell_area_obj, &cell_center_obj))
        return nullptr;

    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nodes = (PyArrayObject *)PyArray_FROM_OTF(nodes_obj, NPY_FLOAT_TYPE, NPY_ARRAY_IN_ARRAY);// read only and c-contiguous
    PyArrayObject *cell_area = (PyArrayObject *)PyArray_FROM_OTF(cell_area_obj, NPY_FLOAT_TYPE, NPY_ARRAY_INOUT_ARRAY); //read, write and c-contiguous
    PyArrayObject *cell_center = (PyArrayObject *)PyArray_FROM_OTF(cell_center_obj, NPY_FLOAT_TYPE, NPY_ARRAY_INOUT_ARRAY);

    const auto free_tables = [&]() {
        Py_XDECREF(cells); cells = nullptr;
        Py_XDECREF(nodes); nodes = nullptr;
        Py_XDECREF(cell_area); cell_area = nullptr;
        Py_XDECREF(cell_center); cell_center = nullptr;
    };

    if (!cells || !nodes || !cell_area || !cell_center) {
        free_tables();
        return nullptr;
    }


    auto const py_cells = PyArray<int32_t, 2>(cells);
    auto py_nodes = PyArray<fdx_t, 2>(nodes);
    auto py_cell_area = PyArray<fdx_t, 1>(cell_area);
    auto py_cell_center = PyArray<fdx_t, 2>(cell_center);

    // call a worker function
    if (dim == 2) {
        compute_cell_center_area_2d(&py_cells, &py_nodes, &py_cell_area, &py_cell_center);
    } else {
        compute_cell_center_volume_3d(&py_cells, &py_nodes, &py_cell_area, &py_cell_center);
    }

    // If a copy was made, write it back
    if (PyArray_ResolveWritebackIfCopy(cell_area) < 0 || PyArray_ResolveWritebackIfCopy(cell_center) < 0) {
        free_tables();
        return nullptr;
    }

    // return NULL
    free_tables();
    return PyFloat_FromDouble(Py_NAN);
}

/**
 * @brief Specific geometric Python binding for 2D cell area and centroid computations.
 * @details Wraps `py_compute_cell_center_area_volume_general` with `dim = 2`.
 * @param self The module object.
 * @param args Python tuple of function arguments.
 * @return PyObject* In-place mutation wrapper.
 */
static PyObject *py_compute_cell_center_area_2d(PyObject *self, PyObject *args) {
    return py_compute_cell_center_area_volume_general(self, args, 2);
}

/**
 * @brief Specific geometric Python binding for 3D cell volume and centroid computations.
 * @details Wraps `py_compute_cell_center_area_volume_general` with `dim = 3`.
 * @param self The module object.
 * @param args Python tuple of function arguments.
 * @return PyObject* In-place mutation wrapper.
 */
static PyObject *py_compute_cell_center_volume_3d(PyObject *self, PyObject *args) {
    return py_compute_cell_center_area_volume_general(self, args, 3);
}

/* -------- module definition --------------------------------------- */
// ----------------- Method Table -----------------------

static const char doc_make_n_part_graph_k_way[] = R"doc(
make_n_part_graph_k_way(graph, nb_part) -> numpy.ndarray

Partition a graph into `nb_part` parts using METIS_PartGraphKway.

Parameters
----------
graph : numpy.ndarray[int32]
    A 2D adjacency matrix of size `(n_vertices, max_cell_neighbors)`.
    The last element of each row contains the node degree (number of neighbors).
nb_part : int
    Number of partitions to create (must be >= 2).

Returns
-------
numpy.ndarray[int32]
    A 1D array of shape `(n_vertices,)` containing the partition ID for each vertex.
)doc";

static const char doc_make_n_part_mesh_dual[] = R"doc(
make_n_part_mesh_dual(cells, nb_parts, n_common) -> numpy.ndarray

Partition a mesh into `nb_parts` parts using the dual graph formulation (METIS_PartMeshDual).
Two elements are considered adjacent if they share at least `n_common` nodes.

Parameters
----------
cells : numpy.ndarray[int32]
    A 2D array of element connectivity of size `(n_elements, max_nodes_per_element)`.
    The last element in each row contains the element degree (number of nodes).
nb_parts : int
    Number of partitions to create (must be >= 2).
n_common : int
    Number of common nodes required to define adjacency between two elements.

Returns
-------
numpy.ndarray[int32]
    A 1D array of shape `(n_elements,)` containing the partition ID for each element.
)doc";

static const char doc_make_n_part_mesh_nodal[] = R"doc(
make_n_part_mesh_nodal(cells, nb_parts) -> numpy.ndarray

Partition a mesh into `nb_parts` parts using the nodal graph formulation (METIS_PartMeshNodal).

Parameters
----------
cells : numpy.ndarray[int32]
    A 2D array of element connectivity of size `(n_elements, max_nodes_per_element)`.
    The last element in each row contains the element degree (number of nodes).
nb_parts : int
    Number of partitions to create (must be >= 2).

Returns
-------
numpy.ndarray[int32]
    A 1D array of shape `(n_elements,)` containing the partition ID for each element.
)doc";


/* ------------------------------------------------------------------------- */
/*  Docstring for py_create_local_domains                                    */
/* ------------------------------------------------------------------------- */
static const char create_local_domains_doc[] = R"doc(
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
part_vert : numpy.ndarray[int32]               (n_vertices,)
    Partition id of every vertex **before** repartitioning.
node_cellid : numpy.ndarray[int32]             (n_vertices,)
    Global cell id that first owns each vertex.
node_phyid : numpy.ndarray[int32]              (n_vertices,)
    Physical (boundary-condition) id attached to each vertex.
cells : numpy.ndarray[int32]                   (n_cells, max_nodes_per_cell)
    Node connectivity of each cell (global node indices).
cells_type : numpy.ndarray[int8]               (n_cells,)
    Element-type code per cell (e.g. 5 = tetra, 9 = hex, ...).
nodes : numpy.ndarray[float32|float64]         (n_vertices, ndim)
    Cartesian coordinates of every node (`z` column present only for 3-D).
phy_faces : numpy.ndarray[int32]               (n_phy_faces, max_nodes_per_face)
    Connectivity of boundary faces (global node indices).
phy_faces_name : numpy.ndarray[int32]          (n_phy_faces,)
    Physical-name / BC id of each boundary face.
nb_parts : int
    Number of sub-domains to create (must be >= 2).
dim : int
    Dimension of the mesh (2 or 3).

Returns
-------
list[tuple]
    A Python list of length `nb_parts`.
    The `p`-th element (`parts[p]`) is a 22-item tuple containing
    every array that belongs to partition `p`.

    0. nodes               - fdx_t (n_nodes_p, ndim)
    1. cells               - int32 (n_cells_p, max_nodes_per_cell)
    2. cells_type          - int8  (n_cells_p,)
    3. phy_faces           - int32 (n_phy_faces_p, max_nodes_per_face)
    4. phy_faces_name      - int32 (n_phy_faces_p,)
    5. cell_loctoglob      - int32 (n_cells_p,)
    6. node_loctoglob      - int32 (n_nodes_p,)
    7. node_oldname        - int32 (n_nodes_p,)
    8. halo_neighsub       - int32 (2, n_neigh_parts_p)
    9. node_halos          - int32 (2 * n_ext_halo_nodes_p,)
   10. halo_halosext       - int32 (n_halos_p, max_cell_nodeid + 2)
   11. halo_halosint       - int32 (n_halos_int_p,)
   12. halo_centvol        - fdx_t (n_halos_p, ndim + 1)
   13. phyid_neighbor      - int32 [[Neighbor partition ID, nb_recv, nb_send] ...]
   14. phyid_recv          - int32 [PhyFaceGlobalId, ...]
   15. phyid_send          - int32 [PhyFaceLocalId], ...
   16. node_halophyid      - int32 [NodeLocalId1, IndexPointToPhyId_recv, ... Size1, NodeLocalId2, ... Size2, ...., SizeN]
   17. cell_halophyid      - int32 [...]
   18. max_node_phyid      - int
   19. max_node_halophyid  - int
   20. max_cell_phyid      - int
   21. max_cell_halophyid  - int

Notes
-----
* All arrays are new NumPy objects; none of the inputs are modified.
* Array dtypes and shapes are guaranteed as shown; callers may rely on them.
)doc";

/* ────────────────────────────────────────────────────────────────────────── */
/*  Docstring: compute_cell_center_area_2d                                   */
/* ────────────────────────────────────────────────────────────────────────── */
static const char cell_center_area_2d_doc[] = R"doc(
compute_cell_center_area_2d(cells, nodes, cell_area, cell_center) -> None

Compute the geometric **area** and **centroid** of every 2-D cell.

Parameters
----------
cells : numpy.ndarray[int32]               (n_cells, n_nodes_per_cell)
    Connectivity table - each row holds the global node indices of one cell.
nodes : numpy.ndarray[float32 | float64]   (n_nodes, 2)
    Cartesian coordinates `[[x, y], ...]` of every vertex.
cell_area : numpy.ndarray[float32 | float64]  (n_cells,)
    **Output (modified in-place)**. Receives the area of each cell.
cell_center : numpy.ndarray[float32 | float64] (n_cells, 2)
    **Output (modified in-place)**. Receives the centroid `[x_c, y_c]` of each cell.

Returns
-------
None
)doc";

/* ────────────────────────────────────────────────────────────────────────── */
/*  Docstring: compute_cell_center_volume_3d                                 */
/* ────────────────────────────────────────────────────────────────────────── */
static const char cell_center_volume_3d_doc[] = R"doc(
compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center) -> None

Compute the geometric **volume** and **centroid** of every 3-D cell.

Parameters
----------
cells : numpy.ndarray[int32]               (n_cells, n_nodes_per_cell)
    Connectivity table - each row holds the global node indices of one cell.
nodes : numpy.ndarray[float32 | float64]   (n_nodes, 3)
    Cartesian coordinates `[[x, y, z], ...]` of every vertex.
cell_volume : numpy.ndarray[float32 | float64]  (n_cells,)
    **Output (modified in-place)**. Receives the volume of each cell.
cell_center : numpy.ndarray[float32 | float64]  (n_cells, 3)
    **Output (modified in-place)**. Receives the centroid `[x_c, y_c, z_c]` of each cell.

Returns
-------
None
)doc";

static PyMethodDef ManapyMethods[] = {
    { "make_n_part_graph_k_way", py_make_n_part_graph_k_way, METH_VARARGS, doc_make_n_part_graph_k_way },
    { "make_n_part_mesh_dual", py_make_n_part_mesh_dual, METH_VARARGS, doc_make_n_part_mesh_dual },
    { "make_n_part_mesh_nodal", py_make_n_part_mesh_nodal, METH_VARARGS, doc_make_n_part_mesh_nodal },
    { "create_local_domains", py_create_local_domains, METH_VARARGS, create_local_domains_doc },
    { "compute_cell_center_area_2d", py_compute_cell_center_area_2d, METH_VARARGS, cell_center_area_2d_doc },
    { "compute_cell_center_volume_3d", py_compute_cell_center_volume_3d, METH_VARARGS, cell_center_volume_3d_doc },
    { NULL, NULL, 0, NULL }
};

// ----------------- Module Definition -----------------------
static struct PyModuleDef manapy_module = {
    PyModuleDef_HEAD_INIT,
    STR(MODULE_NAME),    // dynamic module name
    "Manapy domain partitioning helpers (METIS-backed)",
    -1,
    ManapyMethods
};

// ----------------- Dynamic Init Function -----------------------
#define CONCAT(a, b) a##b
#define EXPAND_AND_CONCAT(a, b) CONCAT(a, b)

#define MAKE_INIT_FUNC(name) EXPAND_AND_CONCAT(PyInit_, name)

//EXPAND to PyInit_manapy_part32_32 Or PyInit_manapy_part32_64 PyInit_manapy_part64_32 PyInit_manapy_part64_64
PyMODINIT_FUNC MAKE_INIT_FUNC(MODULE_NAME)(void)
{
    import_array(); // initialize NumPy C-API
    return PyModule_Create(&manapy_module);
}

