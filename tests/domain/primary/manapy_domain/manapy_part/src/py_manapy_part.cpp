#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#include <numpy/arrayobject.h>
#include "manapy_part.h"

// TODO check leaks

/* Graph representation in compressed Sparse Row (CSR) format
 * graph a 2D int32 numpy array of size (number of cells, max cell neighbors)
 * Example (4 vertices):
 *   [[1, 2, 0, 2],   // v0 → {1,2}
 *    [0, 2, 3, 3],   // v1 → {0,2,3}
 *    [0, 1, 0, 2],   // v2 → {0,1}
 *    [1, 0, 0, 1]]   // v3 → {1}
 * xadj -> is a 1D array of size nvtxs + 1 It tells you where the list of neighbors starts for each vertex in the flat `adjncy` array.
 * adjncy is a 1D array storing all neighbors of all vertices, flattened. Length of adjncy = total number of edges
 * nvtxs -> number of vertecies
 * total_deg -> total degree of the graph. Somme of all edges
 * idx_t is the integer type that METIS uses for every “index-like” quantity— vertex IDs ...
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

    print_instant("Nb_vertices = %ld\n", (long) nb_vertices);
    print_instant("total_deg size = %d\n", (long) deg_sum);
    print_instant("Nb_nodes = %d\n", (long) (*nb_nodes));
    return 0;

}


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

    print_instant("METIS_PartGraphKway");
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

    print_instant("METIS_PartMeshDual");
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

    print_instant("METIS_PartMeshNodal");
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


static PyObject *py_make_n_part_graph_k_way(PyObject *self, PyObject *args) {
    PyObject *graph_obj = nullptr;
    int nb_parts = 0;

    if (!PyArg_ParseTuple(args, "Oi", &graph_obj, &nb_parts))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *graph = (PyArrayObject *)PyArray_FROM_OTF(graph_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!graph)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;

    ret = make_n_part_graph_k_way(graph, nb_parts, &part_vert);
    if (ret == -1) {
        Py_DECREF(graph);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(graph)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, int_type, part_vert);
    if (!part_array) {
        Py_DECREF(graph);
        free(part_array);
        return nullptr;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    Py_DECREF(graph);

    PyObject *ret_data = Py_BuildValue("O", part_array);
    if (!ret_data)
        Py_DECREF(part_array);
    return ret_data;
}

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

    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!cells)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;


    ret = make_n_part_mesh_dual(cells, nb_parts, n_common, &part_vert);
    if (ret == -1) {
        Py_DECREF(cells);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(cells)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, int_type, part_vert);
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

static PyObject *py_make_n_part_mesh_nodal(PyObject *self, PyObject *args) {
    PyObject *cells_obj = nullptr;
    int nb_parts = 0;

    if (!PyArg_ParseTuple(args, "Oi", &cells_obj, &nb_parts))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!cells)
        return nullptr;

    idx_t *part_vert = nullptr;
    idx_t ret;


    ret = make_n_part_mesh_nodal(cells, nb_parts, &part_vert);
    if (ret == -1) {
        Py_DECREF(cells);
        return nullptr;
    }


    const npy_intp dims[1] = { PyArray_DIMS(cells)[0] };
    PyObject *part_array = PyArray_SimpleNewFromData(1, dims, int_type, part_vert);
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

    if (!PyArg_ParseTuple(args, "OOOOOOOOi", &part_vert_obj, &node_cellid_obj, &node_phyid_obj, &cells_obj, &cells_type_obj, &nodes_obj, &phy_faces_obj, &phy_faces_name_obj, &nb_parts))
        return nullptr;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
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
    PyArrayObject *nodes = (PyArrayObject *)PyArray_FROM_OTF(nodes_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
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
        PyArray<double, 2> py_nodes(nodes);
        PyArray<int32_t, 2> py_phy_faces(phy_faces);
        PyArray<int32_t, 1> py_phy_faces_name(phy_faces_name);

        PyObject *py_list_result_tmp = devide(&py_part_vert, &py_node_cellid, &py_nodes, &py_cells, &py_cells_type, &py_phy_faces, &py_phy_faces_name, &py_node_phyid, nb_parts);

        // time_it("");
        // PyObject *py_list_result = create_local_domains(
        //   local_domains,
        //   &py_part_vert,
        //   &py_node_cellid,
        //   &py_node_phyid,
        //   &py_cells,
        //   &py_cells_type,
        //   &py_nodes,
        //   &py_phy_faces,
        //   &py_phy_faces_name,
        //   nb_parts
        // );
        // time_it("create_local_domains");

        // Free resources and return
        free_tables();
        return py_list_result_tmp;
    } catch (std::exception &e) {
        print_instant("%s", e.what());
        PyErr_SetString(PyExc_RuntimeError, e.what());

        // Free resources
        free_tables();
        return nullptr;
    }
}


/* -------- module definition --------------------------------------- */
// ----------------- Method Table -----------------------

static const char *doc_make_n_part_graph_k_way =
    "make_n_part(graph, nb_part) -> np.ndarray\n"
    "\n"
    "Partition a graph into `nb_part` parts using METIS.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "graph : numpy.ndarray\n"
    "    2D adjacency matrix.\n"
    "nb_part : int\n"
    "    Number of partitions.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray\n"
    "    Array of partition IDs for each vertex.";

static const char *doc_make_n_part_mesh_dual =
    "make_n_part(cells, nb_nodes, nb_parts, n_common) -> np.ndarray\n"
    "\n"
    "Partition a mesh into `nb_part` parts using METIS.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "cells : numpy.ndarray\n"
    "nb_parts : int\n"
    "n_common : int\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray\n"
    "    Array of partition IDs for each vertex.";


static const char *doc_make_n_part_mesh_nodal =
    "make_n_part(cells, nb_nodes, nb_parts, n_common) -> np.ndarray\n"
    "\n"
    "Partition a mesh into `nb_part` parts using METIS.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "cells : numpy.ndarray\n"
    "nb_parts : int\n"
    "\n"
    "Returns\n"
    "-------\n"
    "numpy.ndarray\n"
    "    Array of partition IDs for each vertex.";


static PyMethodDef ManapyMethods[] = {
    { "make_n_part_graph_k_way", py_make_n_part_graph_k_way, METH_VARARGS, doc_make_n_part_graph_k_way },
    { "make_n_part_mesh_dual", py_make_n_part_mesh_dual, METH_VARARGS, doc_make_n_part_mesh_dual },
    { "make_n_part_mesh_nodal", py_make_n_part_mesh_nodal, METH_VARARGS, doc_make_n_part_mesh_nodal },
    { "create_local_domains", py_create_local_domains, METH_VARARGS, nullptr },
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
#define INIT_FUNC_NAME(module) PyInit_##module
#define MAKE_INIT_FUNC(module) INIT_FUNC_NAME(module)

PyMODINIT_FUNC MAKE_INIT_FUNC(MODULE_NAME)(void)
{
    import_array();  // initialize NumPy C-API
    return PyModule_Create(&manapy_module);
}



