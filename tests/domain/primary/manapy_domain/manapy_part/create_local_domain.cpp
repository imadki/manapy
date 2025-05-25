#include <iostream>
#include <tuple>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <metis.h>
#include <map>
#include <set>
#include <stdarg.h>

typedef npy_float32 fdx_t;
constexpr int int_type = NPY_INT32;
constexpr int float_type = NPY_FLOAT32;

enum CELL_TYPE {
    Triangle = 1,
    Quad = 2,
    Tetra = 3,
    Hexahedron = 4,
    Pyramid = 5
};

struct LocalDomainStruct {
    PyArrayObject *nodes = nullptr;
    PyArrayObject *cells = nullptr;
    PyArrayObject *cells_type = nullptr;
    PyArrayObject *phy_faces = nullptr;
    PyArrayObject *phy_faces_name = nullptr;
    PyArrayObject *cell_loctoglob = nullptr;
    PyArrayObject *node_loctoglob = nullptr;
    PyArrayObject *halo_neighsub = nullptr;
    PyArrayObject *node_halos = nullptr;
    PyObject *tuple_res = nullptr;
    idx_t max_cell_nodeid = 0;
    idx_t max_cell_faceid = 0;
    idx_t max_face_nodeid = 0;

    // Temporarily
    idx_t max_phy_face_nodeid = 0;
    std::map<idx_t, idx_t> map_cells;
    std::map<idx_t, idx_t> map_phy_faces;
    std::map<idx_t, idx_t> map_nodes;

    LocalDomainStruct() {}

    int create_tables(const idx_t nb_cells, const idx_t nb_nodes, const idx_t nb_phyfaces) {
        const npy_intp l_cells_dim[2] = {nb_cells, this->max_cell_nodeid};
        PyArrayObject *l_cells = (PyArrayObject *)PyArray_SimpleNew(2, l_cells_dim, int_type);

        const npy_intp l_cells_type_dim[1] = {nb_cells};
        PyArrayObject *l_cells_type = (PyArrayObject *)PyArray_SimpleNew(1, l_cells_type_dim, int_type);

        const npy_intp l_cell_loctoglob_dim[1] = {nb_cells};
        PyArrayObject *l_cell_loctoglob = (PyArrayObject *)PyArray_SimpleNew(1, l_cell_loctoglob_dim, int_type);

        const npy_intp l_nodes_dim[2] = {nb_nodes, 3};
        PyArrayObject *l_nodes = (PyArrayObject *)PyArray_SimpleNew(2, l_nodes_dim, float_type);

        const npy_intp l_node_loctoglob_dim[1] = {nb_nodes};
        PyArrayObject *l_node_loctoglob = (PyArrayObject *)PyArray_SimpleNew(1, l_node_loctoglob_dim, int_type);

        const npy_intp l_phy_faces_dim[2] = {nb_phyfaces, this->max_phy_face_nodeid};
        PyArrayObject *l_phy_faces = (PyArrayObject *)PyArray_SimpleNew(2, l_phy_faces_dim, int_type);

        const npy_intp l_phy_faces_name_dim[1] = {nb_phyfaces};
        PyArrayObject *l_phy_faces_name = (PyArrayObject *)PyArray_SimpleNew(1, l_phy_faces_name_dim, int_type);



        if (!l_cells || !l_cells_type || !l_cell_loctoglob || !l_nodes || !l_node_loctoglob || !l_phy_faces || !l_phy_faces_name)
            return -1;

        this->cells = l_cells;
        this->cells_type = l_cells_type;
        this->cell_loctoglob = l_cell_loctoglob;
        this->nodes = l_nodes;
        this->node_loctoglob = l_node_loctoglob;
        this->phy_faces = l_phy_faces;
        this->phy_faces_name = l_phy_faces_name;

        return 0;

    }

    int create_halo_tables(const idx_t halo_neighsub_size, const idx_t node_halos_size) {
        const npy_intp l_halo_neighsub_dim[1] = {halo_neighsub_size};
        PyArrayObject *l_halo_neighsub = (PyArrayObject *)PyArray_SimpleNew(1, l_halo_neighsub_dim, int_type);

        const npy_intp l_node_halos_dim[1] = {node_halos_size};
        PyArrayObject *l_node_halos = (PyArrayObject *)PyArray_SimpleNew(1, l_node_halos_dim, int_type);

        if (!l_halo_neighsub || !l_node_halos)
            return -1;

        this->halo_neighsub = l_halo_neighsub;
        this->node_halos = l_node_halos;

        return 0;
    }

    int create_tuple() {
        PyObject *tuple = Py_BuildValue("(OOOOOOOOO)", this->nodes, this->cells, this->cells_type,
            this->phy_faces, this->phy_faces_name, this->cell_loctoglob, this->node_loctoglob, this->halo_neighsub, this->node_halos);
        if (!tuple)
            return -1;
        // tuple holds references now
        Py_DECREF(this->nodes);
        Py_DECREF(this->cells);
        Py_DECREF(this->cells_type);
        Py_DECREF(this->phy_faces);
        Py_DECREF(this->phy_faces_name);
        Py_DECREF(this->cell_loctoglob);
        Py_DECREF(this->node_loctoglob);
        Py_DECREF(this->halo_neighsub);
        Py_DECREF(this->node_halos);
        this->nodes = nullptr;
        this->cells = nullptr;
        this->cells_type = nullptr;
        this->phy_faces = nullptr;
        this->phy_faces_name = nullptr;
        this->cell_loctoglob = nullptr;
        this->node_loctoglob = nullptr;
        this->halo_neighsub = nullptr;
        this->node_halos = nullptr;

        this->tuple_res = tuple;
        return 0;
    }

    ~LocalDomainStruct() {
        Py_XDECREF(this->nodes);
        Py_XDECREF(this->cells);
        Py_XDECREF(this->cells_type);
        Py_XDECREF(this->phy_faces);
        Py_XDECREF(this->phy_faces_name);
        Py_XDECREF(this->cell_loctoglob);
        Py_XDECREF(this->node_loctoglob);
        Py_XDECREF(this->halo_neighsub);
        Py_XDECREF(this->node_halos);
        Py_XDECREF(this->tuple_res);
    }
};

std::vector<idx_t>    get_max_info(const idx_t cell_type) {
    if (cell_type == CELL_TYPE::Triangle) {
        return {3, 2, 3};
    } else if (cell_type == CELL_TYPE::Quad) {
        return {4, 2, 4};
    } else if (cell_type == CELL_TYPE::Tetra) {
        return {4, 3, 4};
    } else if (cell_type == CELL_TYPE::Hexahedron) {
        return {6, 4, 8};
    } else if (cell_type == CELL_TYPE::Pyramid) {
        return {5, 4, 5};
    }
    return {0, 0, 0};
}

#include <Python.h>
#include <stdarg.h>

void print_instant(const char *fmt, ...) {
    return ;
    char buffer[512];  // temp string buffer
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);  // format the string
    va_end(args);

    // Import sys module and write to sys.stdout
    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) return;

    PyObject *stdout = PyObject_GetAttrString(sys, "stdout");
    if (stdout) {
        PyObject *write_result = PyObject_CallMethod(stdout, "write", "s", buffer);
        Py_XDECREF(write_result);

        PyObject *flush_result = PyObject_CallMethod(stdout, "flush", NULL);
        Py_XDECREF(flush_result);

        Py_DECREF(stdout);
    }

    Py_DECREF(sys);
}



/**
 * Performs binary search on a sorted array of integers.
 *
 * @param array  Pointer to array of integers.
 * @param item   The item to search for.
 * @param size   The size of the array.
 * @return       Index >= 0 if found, else -1.
 */
int binary_search(const idx_t *array, idx_t item, idx_t size) {
    idx_t left = 0;
    idx_t right = size - 1;

    while (left <= right) {
        const idx_t mid = (left + right) / 2;
        const idx_t mid_val = array[mid];

        if (mid_val == item) {
            return mid;
        } else if (mid_val < item) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

void    intersect_nodes(const idx_t *face_nodes, const idx_t nb_face_nodes, PyArrayObject *node_cellid,  idx_t *intersect) {
    idx_t index = 0;

    intersect[0] = -1;
    intersect[1] = -1;
    const npy_intp *dims = PyArray_DIMS(node_cellid);

    const idx_t *cells = (idx_t *)PyArray_GETPTR2(node_cellid, face_nodes[0], 0);
    for (idx_t i = 0; i < cells[dims[1] - 1]; i++) {
        intersect[index] = cells[i];
        for (idx_t j = 1; j < nb_face_nodes; j++) {
            const idx_t *n_cells = (idx_t *)PyArray_GETPTR2(node_cellid, face_nodes[j], 0);
            const idx_t size = n_cells[dims[1] - 1];
            if (binary_search(n_cells, cells[i], size) == -1) {
                intersect[index] = -1;
                break;
            }
        }
        if (intersect[index] != -1)
            index++;
        if (index >= 2)
            return;
    }
}

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
static int dense_to_csr(PyArrayObject *graph, idx_t **xadj, idx_t **adjncy, idx_t *nvtxs, idx_t *total_deg) {
    const npy_intp *dims = PyArray_DIMS(graph);
    const idx_t nb_vertices = (idx_t) dims[0];
    const idx_t max_cols = (idx_t)dims[1];
    if (max_cols < 1) {
        PyErr_SetString(PyExc_ValueError, "graph must have at least one column (size field)");
        return -1;
    }

    // Get total_deg and nvtxs
    idx_t deg_sum = 0;
    print_instant("Nb_vertices = %ld\n", (long) nb_vertices);
    for (idx_t vertex = 0; vertex < nb_vertices; vertex++) {
        const idx_t size = *(idx_t *)PyArray_GETPTR2(graph, vertex, max_cols - 1);
        if (size < 0 || size >= max_cols) {
            PyErr_Format(PyExc_ValueError, "row %ld has invalid size %ld (max_cols=%ld)", vertex,
                         size, (max_cols - 1));
            return -1;
        }
        deg_sum += size;
    }
    print_instant("size = %d\n", (long) deg_sum);
    *total_deg = deg_sum;
    *nvtxs = nb_vertices;

    //Get xadj and adjncy
    *xadj = (idx_t *)malloc(sizeof(idx_t) * (nb_vertices + 1));
    *adjncy = (idx_t *)malloc(sizeof(idx_t) * (deg_sum));
    if (*xadj == nullptr || *adjncy == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }

    idx_t counter = 0;
    (*xadj)[0] = 0;
    for (idx_t i = 0; i < nb_vertices; i++) {
        const idx_t deg = *(idx_t *)PyArray_GETPTR2(graph, i, max_cols - 1);
        for (idx_t j = 0; j < deg; j++) {
            const idx_t nb = *(idx_t *)PyArray_GETPTR2(graph, i, j);
            if (nb < 0 || nb >= *nvtxs) {
                free(*xadj);
                free(*adjncy);
                PyErr_Format(PyExc_ValueError, "row %ld, col %ld has invalid neighbour %ld", i, j, nb);
                return -1;
            }
            (*adjncy)[counter++] = nb;
        }
        (*xadj)[i + 1] = counter;
    }
    return 0;

}

/*
 * If you request 4 parts on a 3-vertex graph, METIS will do its best, but you may see something like:
 * part = [0, 0, 1]  // only 2 parts used
 * If your graph is very small, or
 * Disconnected, or
 * Your balance constraints are weird
*/
static int make_n_part(PyArrayObject *graph, idx_t nb_part, idx_t **part_vert) {
    idx_t *xadj;
    idx_t *adjncy;
    idx_t nvtxs;
    idx_t deg_sum;
    idx_t ret;

    print_instant("Start dense to csr\n");
    ret = dense_to_csr(graph, &xadj, &adjncy, &nvtxs, &deg_sum);
    if (ret < 0)
        return -1;

    print_instant("End dense to csr\n");
    idx_t *part_idx = (idx_t *)malloc(sizeof(idx_t) * nvtxs);
    if (part_idx == nullptr) {
        free(xadj);
        free(adjncy);
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }

    idx_t ncon = 1;
    idx_t edgecut = 0;

    ret = METIS_PartGraphKway(&nvtxs, &ncon,
                                 xadj, adjncy,
                                 nullptr, nullptr, nullptr,
                                 &nb_part,
                                 nullptr, nullptr,
                                 nullptr,
                                 &edgecut, part_idx);
    print_instant("End METIS_PartGraphKway\n");
    free(xadj);
    free(adjncy);

    if (ret != METIS_OK) {
        free(part_idx);
        PyErr_Format(PyExc_RuntimeError, "METIS_PartGraphKway failed (status=%d)", ret);
        return -1;
    }

    // Return
    *part_vert = part_idx;
    print_instant("End function\n");
    return 0;
}

static int  create_sub_domains(PyArrayObject *graph,
            PyArrayObject *node_cellid,
            PyArrayObject *cells,
            PyArrayObject *cells_type,
            PyArrayObject *nodes,
            PyArrayObject *phy_faces,
            PyArrayObject *phy_faces_name,
            idx_t nb_parts,
            LocalDomainStruct *local_domains
            ) {
    idx_t *part_vert = nullptr;
    idx_t ret;

    print_instant("Make n part\n");
    ret = make_n_part(graph, nb_parts, &part_vert);
    if (ret == -1)
        return -1;

    print_instant("Create local cells\n");
    const npy_intp *cells_dim = PyArray_DIMS(cells);
    for (idx_t i = 0; i < cells_dim[0]; i++) {
        print_instant("%d\n", i);
        const idx_t size = *(idx_t *)PyArray_GETPTR2(cells, i, cells_dim[1] - 1);
        const idx_t p = part_vert[i];
        const idx_t cell_type = *(idx_t *)PyArray_GETPTR1(cells_type, i);
        auto max_info = get_max_info(cell_type);

        local_domains[p].max_cell_faceid = std::max(max_info[0], local_domains[p].max_cell_faceid);
        local_domains[p].max_face_nodeid = std::max(max_info[1], local_domains[p].max_face_nodeid);
        local_domains[p].max_cell_nodeid = std::max(max_info[2], local_domains[p].max_cell_nodeid);

        // Create local cells
        local_domains[p].map_cells[i] = local_domains[p].map_cells.size();

        // Create local nodes
        for (idx_t j = 0; j < size; j++) {
            const idx_t node = *(idx_t *)PyArray_GETPTR2(cells, i, j);
            if (local_domains[p].map_nodes.find(node) == local_domains[p].map_nodes.end()) {
                local_domains[p].map_nodes[node] = local_domains[p].map_nodes.size();
            }
        }
    }

    // Create Physical Faces Parts
    idx_t intersect_cell[2];
    idx_t total_nb_phyfaces = 0;
    const npy_intp *phyfaces_dim = PyArray_DIMS(phy_faces);
    print_instant("Create Physical faces\n");
    for (idx_t i = 0; i < phyfaces_dim[0]; i++) {
        const idx_t *phy_face = (idx_t *)PyArray_GETPTR2(phy_faces, i, 0);
        const idx_t size = phy_face[phyfaces_dim[1] - 1];
        intersect_nodes(phy_face, size, node_cellid, intersect_cell);
        if (intersect_cell[0] != -1) {
            const idx_t p = part_vert[intersect_cell[0]];
            local_domains[p].max_phy_face_nodeid = std::max(size, local_domains[p].max_phy_face_nodeid);
            local_domains[p].map_phy_faces[i] = local_domains[p].map_phy_faces.size();
            total_nb_phyfaces++;
        }
    }
    if (total_nb_phyfaces != phyfaces_dim[0]) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Warning: not all the physical faces match the domain faces !! %d "
            "where the number of physical faces is %ld",
            total_nb_phyfaces, phyfaces_dim[0]);

        PyErr_WarnEx(PyExc_UserWarning, msg, 1);
    }


    // Create Local Domains
    print_instant("Create local domains struct\n");
    for (idx_t p = 0; p < nb_parts; p++) {
        auto &map_cells = local_domains[p].map_cells;
        auto &map_phy_faces = local_domains[p].map_phy_faces;
        auto &map_nodes = local_domains[p].map_nodes;

        const idx_t nb_cells = map_cells.size();
        const idx_t nb_phyfaces = map_phy_faces.size();
        const idx_t nb_nodes = map_nodes.size();
        const idx_t max_cell_nodeid = local_domains[p].max_cell_nodeid;
        const idx_t max_phy_face_nodeid = local_domains[p].max_phy_face_nodeid;


        ret = local_domains[p].create_tables(nb_cells, nb_nodes, nb_phyfaces);
        if (ret == -1) {
            free(part_vert);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate NumPy array");
            return -1;
        }


        idx_t *l_cells_data = (idx_t *)PyArray_DATA(local_domains[p].cells);
        idx_t *l_cells_type_data = (idx_t *)PyArray_DATA(local_domains[p].cells_type);
        idx_t *l_cell_loctoglob_data = (idx_t *)PyArray_DATA(local_domains[p].cell_loctoglob);
        idx_t *l_node_loctoglob_data = (idx_t *)PyArray_DATA(local_domains[p].node_loctoglob);
        fdx_t *l_nodes_data = (fdx_t *)PyArray_DATA(local_domains[p].nodes);
        idx_t *l_phy_faces_data = (idx_t *)PyArray_DATA(local_domains[p].phy_faces);
        idx_t *l_phy_faces_name_data = (idx_t *)PyArray_DATA(local_domains[p].phy_faces_name);
        std::set<idx_t> halo_neighsub;
        std::vector<idx_t> node_halos;

        // # Cells, CellsType, CellsLocToGlob
        print_instant("Cells, CellsType, CellsLocToGlob\n");
        for (auto iter = map_cells.begin(); iter != map_cells.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            // print_instant("%d %d %d\n", nb_cells, nb_nodes, nb_phyfaces)
            // print_instant("%d %d\n", k, local_index);
            l_cells_type_data[local_index] = *(idx_t *)PyArray_GETPTR1(cells_type, k);
            l_cell_loctoglob_data[local_index] = k;

            //copy cell nodes

            const idx_t size = *(idx_t *)PyArray_GETPTR2(cells, k, cells_dim[1] - 1);
            for (idx_t i = 0; i < size; i++) {
                print_instant("->%d %d %d %d %d %d %ld %ld\n", i, local_index, k, size, max_cell_nodeid, nb_cells, cells_dim[0], cells_dim[1]);
                const idx_t g_node = *(idx_t *)PyArray_GETPTR2(cells, k, i);
                l_cells_data[local_index * max_cell_nodeid + i] = map_nodes[g_node];
            }
        }

        // # Nodes, NodesLocToGlob, HaloNeighSub, NodeHalos
        print_instant("Nodes, NodesLocToGlob, HaloNeighSub, NodeHalos\n");
        for (auto iter = map_nodes.begin(); iter != map_nodes.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            l_node_loctoglob_data[local_index] = k;

            for (idx_t i = 0; i < 3; i++) {
                l_nodes_data[local_index * 3 + i] = *(fdx_t *)PyArray_GETPTR2(nodes, k, i);
            }

            // # Halos
            const npy_intp *dims = PyArray_DIMS(node_cellid);
            const idx_t size = *(idx_t *)PyArray_GETPTR2(node_cellid, k, dims[1] - 1);
            for (idx_t i = 0; i < size; i++) {
                const idx_t neighbor_cell = *(idx_t *)PyArray_GETPTR2(node_cellid, k, i);
                const idx_t neighbor_part = part_vert[neighbor_cell];
                if (p != neighbor_cell) {
                    halo_neighsub.insert(neighbor_part);
                    node_halos.push_back(local_index);
                    node_halos.push_back(neighbor_cell);
                }
            }
        }

        //# PhyFaces, PhyFacesName
        print_instant("PhyFaces, PhyFacesName\n");
        for (auto iter = map_phy_faces.begin(); iter != map_phy_faces.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            l_phy_faces_name_data[local_index] = *(idx_t *)PyArray_GETPTR1(phy_faces_name, k);
            const idx_t size = *(idx_t *)PyArray_GETPTR2(phy_faces, k, phyfaces_dim[1] - 1);
            for (idx_t i = 0; i < size; i++) {
                l_phy_faces_data[local_index * max_phy_face_nodeid + i] = *(idx_t *)PyArray_GETPTR2(phy_faces, k, i);
            }
        }

        // # Halo neighsub, Node_halos_data
        print_instant("Halo neighsub, Node_halos_data\n");
        ret = local_domains[p].create_halo_tables((idx_t)halo_neighsub.size(), (idx_t)node_halos.size());
        if (ret == -1) {
            free(part_vert);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate NumPy array");
            return -1;
        }


        idx_t *l_halo_neighsub_data = (idx_t *)PyArray_DATA(local_domains[p].halo_neighsub);
        idx_t *l_node_halos_data = (idx_t *)PyArray_DATA(local_domains[p].node_halos);

        idx_t counter = 0;
        for (auto iter = halo_neighsub.begin(); iter != halo_neighsub.end(); ++iter) {
            l_halo_neighsub_data[counter] = *iter;
            counter++;
        }

        counter = 0;
        for (auto iter = node_halos.begin(); iter != node_halos.end(); ++iter) {
            l_node_halos_data[counter] = *iter;
            counter++;
        }



    }

    print_instant("End function\n");
    free(part_vert);
    return 0;
}

static PyObject *py_make_n_part(PyObject *self, PyObject *args) {
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

    print_instant("Start Creating graph \n");
    ret = make_n_part(graph, nb_parts, &part_vert);
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

static PyObject *py_create_sub_domains(PyObject *self, PyObject *args) {
    // TODO check numpy type and dimension
    PyObject *graph_obj = nullptr;
    PyObject *node_cellid_obj = nullptr;
    PyObject *cells_obj = nullptr;
    PyObject *cells_type_obj = nullptr;
    PyObject *nodes_obj = nullptr;
    PyObject *phy_faces_obj = nullptr;
    PyObject *phy_faces_name_obj = nullptr;
    idx_t nb_part = 0;

    print_instant("Parse Input \n");
    if (!PyArg_ParseTuple(args, "OOOOOOOi", &graph_obj, &node_cellid_obj, &cells_obj, &cells_type_obj, &nodes_obj, &phy_faces_obj, &phy_faces_name_obj, &nb_part))
        return nullptr;
    if (nb_part < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    PyArrayObject *graph = (PyArrayObject *)PyArray_FROM_OTF(graph_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_cellid = (PyArrayObject *)PyArray_FROM_OTF(node_cellid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells_type = (PyArrayObject *)PyArray_FROM_OTF(cells_type_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nodes = (PyArrayObject *)PyArray_FROM_OTF(nodes_obj, float_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces_name = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_name_obj, int_type, NPY_ARRAY_IN_ARRAY);

    if (!graph || !node_cellid || !cells || !cells_type || !nodes || !phy_faces || !phy_faces_name) {
        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        return nullptr;
    }
    if (
        PyArray_NDIM(graph) != 2 || PyArray_TYPE(graph) != int_type || !PyArray_ISCONTIGUOUS(graph) ||
        PyArray_NDIM(node_cellid) != 2 || PyArray_TYPE(node_cellid) != int_type || !PyArray_ISCONTIGUOUS(node_cellid) ||
        PyArray_NDIM(cells) != 2 || PyArray_TYPE(cells) != int_type || !PyArray_ISCONTIGUOUS(cells) ||
        PyArray_NDIM(cells_type) != 1 || PyArray_TYPE(cells_type) != int_type || !PyArray_ISCONTIGUOUS(cells_type) ||
        PyArray_NDIM(nodes) != 2 || PyArray_TYPE(nodes) != float_type || !PyArray_ISCONTIGUOUS(nodes) ||
        PyArray_NDIM(phy_faces) != 2 || PyArray_TYPE(phy_faces) != int_type || !PyArray_ISCONTIGUOUS(phy_faces) ||
        PyArray_NDIM(phy_faces_name) != 1 || PyArray_TYPE(phy_faces_name) != int_type || !PyArray_ISCONTIGUOUS(phy_faces_name)
        ) {
        PyErr_SetString(PyExc_TypeError, "Input Data Type Error");
        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        return nullptr;
    }


    print_instant("Start Creating SubDomains \n");
    LocalDomainStruct *local_domains = new LocalDomainStruct[nb_part];
    //TODO
    if (!local_domains ||
        create_sub_domains(graph, node_cellid, cells, cells_type, nodes, phy_faces, phy_faces_name, nb_part, local_domains) == -1
        ) {

        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        if (!local_domains)
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate local domain struct");
        free(local_domains);
        return nullptr;
    }
    // No need of those
    Py_XDECREF(graph);
    Py_XDECREF(node_cellid);
    Py_XDECREF(cells);
    Py_XDECREF(cells_type);
    Py_XDECREF(nodes);
    Py_XDECREF(phy_faces);
    Py_XDECREF(phy_faces_name);

    PyObject *py_list_result = PyList_New(nb_part);
    if (!py_list_result) {
        free(local_domains);
        return nullptr;
    }

    print_instant("Return List \n");
    for (int i = 0; i < nb_part; i++) {
        if (local_domains[i].create_tuple() == -1) {
            Py_XDECREF(py_list_result);
            free(local_domains);
            return nullptr;
        }
    }


    for (int i = 0; i < nb_part; i++) {
        PyList_SET_ITEM(py_list_result, i, local_domains[i].tuple_res);
        //the ownership transferred to the list.
        local_domains[i].tuple_res = nullptr;
    }

    print_instant("Done \n");
    delete[] local_domains;
    print_instant("Done !\n");
    return py_list_result;
}

/* -------- module definition --------------------------------------- */
static PyMethodDef ManapyMethods[] = {
    { "make_n_part", py_make_n_part, METH_VARARGS, nullptr },
    { "create_sub_domains", py_create_sub_domains, METH_VARARGS, nullptr },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef manapy_module = {
    PyModuleDef_HEAD_INIT,
    "manapy_domain",  /* m_name */
    "Manapy domain partitioning helpers (METIS-backed)", /* m_doc */
    -1,                /* m_size */
    ManapyMethods      /* m_methods */
};

PyMODINIT_FUNC PyInit_manapy_domain(void)
{
    import_array(); /* initialise NumPy C-API */
    return PyModule_Create(&manapy_module);
}

