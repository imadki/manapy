#include "manapy_part.h"

struct LocalDomainStruct {
    PyArrayObject *nodes = nullptr;
    PyArrayObject *cells = nullptr;
    PyArrayObject *cells_type = nullptr;
    PyArrayObject *phy_faces = nullptr;
    PyArrayObject *phy_faces_name = nullptr;
    PyArrayObject *phy_faces_loctoglob = nullptr;
    PyArrayObject *bf_cellid = nullptr;
    PyArrayObject *cell_loctoglob = nullptr;
    PyArrayObject *node_loctoglob = nullptr;
    PyArrayObject *halo_neighsub = nullptr;
    PyArrayObject *node_halos = nullptr;
    PyArrayObject *node_halobfid = nullptr;
    PyArrayObject *shared_bf_recv = nullptr;
    PyArrayObject *bf_recv_part_size = nullptr;
    PyArrayObject *shared_bf_send = nullptr;
    PyArrayObject *halo_halosext = nullptr;
    PyObject *tuple_res = nullptr;
    idx_t max_cell_nodeid = 0;
    idx_t max_cell_faceid = 0;
    idx_t max_face_nodeid = 0;
    idx_t max_node_haloid = 0;
    idx_t max_cell_halofid = 0;
    idx_t max_cell_halonid = 0;

    // Temporarily
    idx_t max_node_halophyid = 0;
    idx_t max_phy_face_nodeid = 0;
    idx_t nb_node_halos = 0;
    std::map<idx_t, idx_t> map_cells;
    std::map<idx_t, idx_t> map_phy_faces;
    std::map<idx_t, idx_t> map_nodes;
    std::map<idx_t, idx_t> map_halos; // halo_g_index -> local_index
    std::set<idx_t> set_bf_recv;
    std::map<idx_t, idx_t> map_bf_recv;
    std::vector<idx_t> vec_shared_bf_recv;
    std::vector<idx_t> vec_shared_bf_send;
    std::set<idx_t> set_halo_bf_neighsub;
    std::set<idx_t> set_halo_neighsub;

    LocalDomainStruct() {}

    int create_tables(const idx_t node_dim) {

        const idx_t nb_cells = this->map_cells.size();
        const idx_t nb_nodes = this->map_nodes.size();
        const idx_t nb_phyfaces = this->map_phy_faces.size();
        const idx_t nb_halos = this->map_halos.size();
        const idx_t nb_halo_neighsub = this->set_halo_neighsub.size();

        print_instant("Create Tables\n");
        const npy_intp l_cells_dim[2] = {nb_cells, this->max_cell_nodeid + 1};
        PyArrayObject *l_cells = (PyArrayObject *)PyArray_SimpleNew(2, l_cells_dim, int_type);

        const npy_intp l_cells_type_dim[1] = {nb_cells};
        PyArrayObject *l_cells_type = (PyArrayObject *)PyArray_SimpleNew(1, l_cells_type_dim, int_type);

        const npy_intp l_cell_loctoglob_dim[1] = {nb_cells};
        PyArrayObject *l_cell_loctoglob = (PyArrayObject *)PyArray_SimpleNew(1, l_cell_loctoglob_dim, int_type);

        const npy_intp l_nodes_dim[2] = {nb_nodes, node_dim};
        PyArrayObject *l_nodes = (PyArrayObject *)PyArray_SimpleNew(2, l_nodes_dim, float_type);

        const npy_intp l_node_loctoglob_dim[1] = {nb_nodes};
        PyArrayObject *l_node_loctoglob = (PyArrayObject *)PyArray_SimpleNew(1, l_node_loctoglob_dim, int_type);

        const npy_intp l_phy_faces_dim[2] = {nb_phyfaces, this->max_phy_face_nodeid + 1};
        PyArrayObject *l_phy_faces = (PyArrayObject *)PyArray_SimpleNew(2, l_phy_faces_dim, int_type);

        const npy_intp l_phy_faces_name_dim[1] = {nb_phyfaces};
        PyArrayObject *l_phy_faces_name = (PyArrayObject *)PyArray_SimpleNew(1, l_phy_faces_name_dim, int_type);

        const npy_intp l_phy_faces_loctoglob_dim[1] = {nb_phyfaces};
        PyArrayObject *l_phy_faces_loctoglob = (PyArrayObject *)PyArray_SimpleNew(1, l_phy_faces_loctoglob_dim, int_type);

        const npy_intp l_bf_cellid_dim[2] = {nb_phyfaces, 2};
        PyArrayObject *l_bf_cellid = (PyArrayObject *)PyArray_SimpleNew(2, l_bf_cellid_dim, int_type);

        const npy_intp l_halo_neighsub_dim[1] = {nb_halo_neighsub};
        PyArrayObject *l_halo_neighsub = (PyArrayObject *)PyArray_SimpleNew(1, l_halo_neighsub_dim, int_type);

        const npy_intp l_node_halos_dim[1] = {this->nb_node_halos};
        PyArrayObject *l_node_halos = (PyArrayObject *)PyArray_SimpleNew(1, l_node_halos_dim, int_type);

        const npy_intp l_node_halobfid_dim[2] = {nb_nodes, this->max_node_halophyid + 1};
        PyArrayObject *l_node_halobfid = (PyArrayObject *)PyArray_ZEROS(2, l_node_halobfid_dim, int_type, 0);

        const npy_intp l_shared_bf_recv_dim[1] = {(idx_t)this->set_bf_recv.size()};
        PyArrayObject *l_shared_bf_recv = (PyArrayObject *)PyArray_SimpleNew(1, l_shared_bf_recv_dim, int_type);

        const npy_intp l_bf_recv_part_size_dim[1] = {(idx_t)this->set_halo_bf_neighsub.size() * 2+2};
        PyArrayObject *l_bf_recv_part_size = (PyArrayObject *)PyArray_SimpleNew(1, l_bf_recv_part_size_dim, int_type);

        const npy_intp l_halo_halosext_dim[2] = {nb_halos, this->max_cell_nodeid + 2};
        PyArrayObject *l_halo_halosext = (PyArrayObject *)PyArray_SimpleNew(2, l_halo_halosext_dim, int_type);


        if (!l_cells || !l_cells_type || !l_cell_loctoglob || !l_nodes || !l_node_loctoglob || !l_phy_faces || !l_phy_faces_name || !l_phy_faces_loctoglob || !l_bf_cellid || !l_halo_neighsub || !l_node_halos || !l_node_halobfid || !l_shared_bf_recv || !l_bf_recv_part_size || !l_halo_halosext)
            return -1;

        this->cells = l_cells;
        this->cells_type = l_cells_type;
        this->cell_loctoglob = l_cell_loctoglob;
        this->nodes = l_nodes;
        this->node_loctoglob = l_node_loctoglob;
        this->phy_faces = l_phy_faces;
        this->phy_faces_name = l_phy_faces_name;
        this->phy_faces_loctoglob = l_phy_faces_loctoglob;
        this->bf_cellid = l_bf_cellid;
        this->halo_neighsub = l_halo_neighsub;
        this->node_halos = l_node_halos;
        this->node_halobfid = l_node_halobfid;
        this->shared_bf_recv = l_shared_bf_recv;
        this->bf_recv_part_size = l_bf_recv_part_size;
        this->halo_halosext = l_halo_halosext;

        print_instant("End Create Tables\n");
        return 0;
    }

    int create_shared_bf_send() {

        const npy_intp l_shared_bf_send_dim[1] = {(idx_t)this->vec_shared_bf_send.size()};
        PyArrayObject *l_shared_bf_send = (PyArrayObject *)PyArray_SimpleNew(1, l_shared_bf_send_dim, int_type);

        if (l_shared_bf_send == nullptr)
            return -1;

        idx_t *l_node_loctoglob_data = (idx_t *)PyArray_DATA(l_shared_bf_send);

        idx_t counter = 0;
        for (auto &item: this->vec_shared_bf_send) {
            l_node_loctoglob_data[counter++] = item;
        }

        this->shared_bf_send = l_shared_bf_send;
        return 0;
    }

    int create_tuple() {
        PyObject *tuple = Py_BuildValue("(OOOOOOOOOOOOOOOOiiiiii)", this->nodes, this->cells, this->cells_type,
            this->phy_faces, this->phy_faces_name, this->phy_faces_loctoglob, this->bf_cellid, this->cell_loctoglob, this->node_loctoglob,
            this->halo_neighsub, this->node_halos, this->node_halobfid, this->shared_bf_recv, this->bf_recv_part_size, this->shared_bf_send, this->halo_halosext,
            this->max_cell_nodeid, this->max_cell_faceid, this->max_face_nodeid, this->max_node_haloid, this->max_cell_halofid, this->max_cell_halonid);
        if (!tuple)
            return -1;
        // tuple holds references now
        Py_DECREF(this->nodes);
        Py_DECREF(this->cells);
        Py_DECREF(this->cells_type);
        Py_DECREF(this->phy_faces);
        Py_DECREF(this->phy_faces_name);
        Py_DECREF(this->phy_faces_loctoglob);
        Py_DECREF(this->bf_cellid);
        Py_DECREF(this->cell_loctoglob);
        Py_DECREF(this->node_loctoglob);
        Py_DECREF(this->halo_neighsub);
        Py_DECREF(this->node_halos);
        Py_DECREF(this->node_halobfid);
        Py_DECREF(this->shared_bf_recv);
        Py_DECREF(this->bf_recv_part_size);
        Py_DECREF(this->shared_bf_send);
        Py_DECREF(this->halo_halosext);
        this->nodes = nullptr;
        this->cells = nullptr;
        this->cells_type = nullptr;
        this->phy_faces = nullptr;
        this->phy_faces_name = nullptr;
        this->phy_faces_loctoglob = nullptr;
        this->bf_cellid = nullptr;
        this->cell_loctoglob = nullptr;
        this->node_loctoglob = nullptr;
        this->halo_neighsub = nullptr;
        this->node_halos = nullptr;
        this->node_halobfid = nullptr;
        this->shared_bf_recv = nullptr;
        this->bf_recv_part_size = nullptr;
        this->shared_bf_send = nullptr;
        this->halo_halosext = nullptr;

        this->tuple_res = tuple;
        return 0;
    }

    ~LocalDomainStruct() {
        Py_XDECREF(this->nodes);
        Py_XDECREF(this->cells);
        Py_XDECREF(this->cells_type);
        Py_XDECREF(this->phy_faces);
        Py_XDECREF(this->phy_faces_name);
        Py_XDECREF(this->phy_faces_loctoglob);
        Py_XDECREF(this->bf_cellid);
        Py_XDECREF(this->cell_loctoglob);
        Py_XDECREF(this->node_loctoglob);
        Py_XDECREF(this->halo_neighsub);
        Py_XDECREF(this->node_halos);
        Py_XDECREF(this->node_halobfid);
        Py_XDECREF(this->shared_bf_recv);
        Py_XDECREF(this->bf_recv_part_size);
        Py_XDECREF(this->shared_bf_send);
        Py_XDECREF(this->halo_halosext);
        Py_XDECREF(this->tuple_res);

        this->nodes = nullptr;
        this->cells = nullptr;
        this->cells_type = nullptr;
        this->phy_faces = nullptr;
        this->phy_faces_name = nullptr;
        this->phy_faces_loctoglob = nullptr;
        this->bf_cellid = nullptr;
        this->cell_loctoglob = nullptr;
        this->node_loctoglob = nullptr;
        this->halo_neighsub = nullptr;
        this->node_halos = nullptr;
        this->node_halobfid = nullptr;
        this->shared_bf_recv = nullptr;
        this->bf_recv_part_size = nullptr;
        this->shared_bf_send = nullptr;
        this->halo_halosext = nullptr;
        this->tuple_res = nullptr;
    }
};

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
int make_n_part(PyArrayObject *graph, idx_t nb_part, idx_t **part_vert) {
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


int  create_sub_domains(PyArrayObject *graph,
            PyArrayObject *node_cellid,
            PyArrayObject *node_bfid,
            PyArrayObject *bf_cellid,
            PyArrayObject *cells,
            PyArrayObject *cell_cellfid,
            PyArrayObject *cell_cellnid,
            PyArrayObject *cells_type,
            PyArrayObject *nodes,
            PyArrayObject *phy_faces,
            PyArrayObject *phy_faces_name,
            idx_t nb_parts,
            LocalDomainStruct *local_domains
            ) {
    idx_t *part_vert = nullptr;
    idx_t ret;

    // Allocate phy_part_vert
    const idx_t nb_phy_faces = PyArray_DIMS(phy_faces)[0];

    // Make n part
    print_instant("Make n part\n");
    ret = make_n_part(graph, nb_parts, &part_vert);
    if (ret == -1) {
        return -1;
    }

    print_instant("Create local cells\n");
    const npy_intp *cells_dim = PyArray_DIMS(cells);

#pragma region Create local cells and nodes, map_halos
    for (idx_t i = 0; i < cells_dim[0]; i++) {
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

        // Determine max_cell_halofid
        const npy_intp *dimfid = PyArray_DIMS(cell_cellfid);
        const idx_t sizefid = *(idx_t *)PyArray_GETPTR2(cell_cellfid, i, dimfid[1] - 1);
        idx_t counterfid = 0;
        for (idx_t j = 0; j < sizefid; j++) {
            const idx_t cellfid = *(idx_t *)PyArray_GETPTR2(cell_cellfid, i, j);
            if (p != part_vert[cellfid])
                counterfid++;
        }
        local_domains[p].max_cell_halofid = std::max(counterfid, local_domains[p].max_cell_halofid);

        // Determine max_cell_halonid, Create HaloMap, Create HaloNeighDomain
        const npy_intp *dimnid = PyArray_DIMS(cell_cellnid);
        const idx_t sizenid = *(idx_t *)PyArray_GETPTR2(cell_cellnid, i, dimnid[1] - 1);
        idx_t counternid = 0;
        for (idx_t j = 0; j < sizenid; j++) {
            const idx_t cellnid = *(idx_t *)PyArray_GETPTR2(cell_cellnid, i, j);
            const idx_t part_cellnid = part_vert[cellnid];
            if (p != part_cellnid) {
                counternid++;
                if (local_domains[p].map_halos.find(cellnid) == local_domains[p].map_halos.end()) {
                    local_domains[p].map_halos[cellnid] = local_domains[p].map_halos.size();
                }
                local_domains[p].set_halo_neighsub.insert(part_cellnid);
            }

        }
        local_domains[p].max_cell_halonid = std::max(counternid, local_domains[p].max_cell_halonid);


    }
#pragma endregion

#pragma region Create Phyical Faces Parts
//Create Physical Faces Parts
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
#pragma endregion

    // =================================
    // Create part_bf
    // =================================
    idx_t size_bf_cellid = (idx_t)PyArray_DIMS(bf_cellid)[0];
    idx_t *part_bf = new (std::nothrow) idx_t[size_bf_cellid];
    if (part_bf == nullptr) {
        free(part_vert);
        return -1;
    }
    for (idx_t i = 0; i < size_bf_cellid; i++) {
        const idx_t cell_id = *(idx_t *)PyArray_GETPTR2(bf_cellid, i, 0);
        part_bf[i] = part_vert[cell_id];
    }

    // Create Local Domains
    print_instant("Create local domains struct\n");
    for (idx_t p = 0; p < nb_parts; p++) {
        print_instant("Create local domains struct p=%d\n", p);
        auto &map_cells = local_domains[p].map_cells;
        auto &map_phy_faces = local_domains[p].map_phy_faces;
        auto &map_nodes = local_domains[p].map_nodes;
        auto &map_halos = local_domains[p].map_halos;
        auto &set_halo_neighsub = local_domains[p].set_halo_neighsub;
        auto &set_halo_bf_neighsub = local_domains[p].set_halo_bf_neighsub;


        const idx_t max_cell_nodeid = local_domains[p].max_cell_nodeid;
        const idx_t max_phy_face_nodeid = local_domains[p].max_phy_face_nodeid;
        auto &set_bf_recv = local_domains[p].set_bf_recv;


        // Calculate local_domains[p].nb_node_halos
        for (auto iter = map_nodes.begin(); iter != map_nodes.end(); ++iter) {
            const idx_t k = iter->first; //global index

            const npy_intp *dims = PyArray_DIMS(node_cellid);
            const idx_t size = *(idx_t *)PyArray_GETPTR2(node_cellid, k, dims[1] - 1);
            bool has_halos = false;
            for (idx_t j = 0; j < size; j++) {
                const idx_t neighbor_cell = *(idx_t *)PyArray_GETPTR2(node_cellid, k, j);
                const idx_t neighbor_part = part_vert[neighbor_cell];
                if (p != neighbor_part) {
                    local_domains[p].nb_node_halos++;
                    has_halos = true;
                }
            }
            if (has_halos) {
                local_domains[p].nb_node_halos += 2; // the node and its counter
            }


            // node halo physical faces
            const npy_intp *node_bfid_dims = PyArray_DIMS(node_bfid);
            const idx_t node_bfid_size = *(idx_t *)PyArray_GETPTR2(node_bfid, k, node_bfid_dims[1] - 1);
            idx_t nb_node_halobf = 0;
            for (idx_t j = 0; j < node_bfid_size; j++) {
                const idx_t neighbor_bf = *(idx_t *)PyArray_GETPTR2(node_bfid, k, j);
                const idx_t neighbor_part = part_bf[neighbor_bf];

                // for every local node collect all neighbor boundary faces
                if (set_bf_recv.find(neighbor_bf) == set_bf_recv.end())
                    set_bf_recv.insert(neighbor_bf);
                // count halo boundary faces
                // determine boundary faces neighsub
                if (p != neighbor_part) {
                    nb_node_halobf++;
                    set_halo_bf_neighsub.insert(neighbor_part);
                }
            }
            local_domains[p].max_node_halophyid = std::max(nb_node_halobf, local_domains[p].max_node_halophyid);

        }
// ++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++

        print_instant("Create map_bf_recv, shared_bf_recv\n");
        // Create map_bf_recv, shared_bf_recv
        local_domains[p].vec_shared_bf_recv = std::vector<idx_t>(set_bf_recv.begin(), set_bf_recv.end());
        std::vector<idx_t> &vec_shared_bf_recv = local_domains[p].vec_shared_bf_recv;
        std::sort(vec_shared_bf_recv.begin(), vec_shared_bf_recv.end(), [&part_bf](const idx_t a, const idx_t b) {
            return part_bf[a] < part_bf[b];
        });

        std::map<idx_t, idx_t> &map_bf_recv = local_domains[p].map_bf_recv;

        for (size_t i = 0; i < vec_shared_bf_recv.size(); i++) {
            map_bf_recv[vec_shared_bf_recv[i]] = (idx_t)i;
        }

// ++++++++++++++++++++++++++++++++++++++
        // Create tables
// ++++++++++++++++++++++++++++++++++++++
        const npy_intp *nodes_dim = PyArray_DIMS(nodes); // 3 or 2
        ret = local_domains[p].create_tables((idx_t)nodes_dim[1]);
        if (ret == -1) {
            free(part_vert);
            free(part_bf);
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
        idx_t *l_phy_faces_loctoglob_data = (idx_t *)PyArray_DATA(local_domains[p].phy_faces_loctoglob);
        idx_t *l_bf_cellid_data = (idx_t *)PyArray_DATA(local_domains[p].bf_cellid);
        idx_t *l_halo_neighsub_data = (idx_t *)PyArray_DATA(local_domains[p].halo_neighsub);
        idx_t *l_node_halos_data = (idx_t *)PyArray_DATA(local_domains[p].node_halos);
        idx_t *l_shared_bf_recv_data = (idx_t *)PyArray_DATA(local_domains[p].shared_bf_recv);
        idx_t *l_bf_recv_part_size_data = (idx_t *)PyArray_DATA(local_domains[p].bf_recv_part_size);
        idx_t *l_halo_halosext_data = (idx_t *)PyArray_DATA(local_domains[p].halo_halosext);


        // # Cells, CellsType, CellsLocToGlob
        print_instant("Cells, CellsType, CellsLocToGlob\n");
        for (auto iter = map_cells.begin(); iter != map_cells.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            l_cells_type_data[local_index] = *(idx_t *)PyArray_GETPTR1(cells_type, k);
            l_cell_loctoglob_data[local_index] = k;

            //copy cell nodes

            const idx_t size = *(idx_t *)PyArray_GETPTR2(cells, k, cells_dim[1] - 1);
            for (idx_t i = 0; i < size; i++) {
                const idx_t g_node = *(idx_t *)PyArray_GETPTR2(cells, k, i);
                l_cells_data[local_index * (max_cell_nodeid + 1) + i] = map_nodes[g_node];
            }
            // size
            l_cells_data[local_index * (max_cell_nodeid + 1) + max_cell_nodeid] = size;
        }

        // # Nodes, NodesLocToGlob, NodeHalos
        idx_t halos_counter = 0;
        for (auto iter = map_nodes.begin(); iter != map_nodes.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            // Create NodesLocToGlob
            l_node_loctoglob_data[local_index] = k;

            // Create Local Nodes Vertices
            const idx_t v_size = nodes_dim[1];
            for (idx_t i = 0; i < v_size; i++) {
                l_nodes_data[local_index * v_size + i] = *(fdx_t *)PyArray_GETPTR2(nodes, k, i);
            }

            // # Calculate max_node_haloid, Create Node neighboring halos
            const npy_intp *dims = PyArray_DIMS(node_cellid);
            const idx_t size = *(idx_t *)PyArray_GETPTR2(node_cellid, k, dims[1] - 1);
            idx_t *node_counter_pointer = nullptr;
            for (idx_t i = 0; i < size; i++) {
                const idx_t neighbor_cell = *(idx_t *)PyArray_GETPTR2(node_cellid, k, i);
                const idx_t neighbor_part = part_vert[neighbor_cell];
                if (p != neighbor_part) {
                    if (node_counter_pointer == nullptr) {
                        l_node_halos_data[halos_counter] = local_index;
                        l_node_halos_data[halos_counter + 1] = 0;
                        node_counter_pointer = &l_node_halos_data[halos_counter + 1];
                        halos_counter += 2;
                    }
                    l_node_halos_data[halos_counter] = map_halos[neighbor_cell];
                    halos_counter++;
                    (*node_counter_pointer)++;
                }
            }
            if (node_counter_pointer) {
                local_domains[p].max_node_haloid = std::max(local_domains[p].max_node_haloid, *node_counter_pointer);
            }


            // node halo boundary face
            PyArrayObject *node_halobfid = local_domains[p].node_halobfid;
            const npy_intp *node_halobfid_dims = PyArray_DIMS(node_halobfid);

            const npy_intp *node_bfid_dims = PyArray_DIMS(node_bfid);
            const idx_t node_bfid_size = *(idx_t *)PyArray_GETPTR2(node_bfid, k, node_bfid_dims[1] - 1);

            for (idx_t j = 0; j < node_bfid_size; j++) {
                const idx_t neighbor_bf = *(idx_t *)PyArray_GETPTR2(node_bfid, k, j);
                const idx_t neighbor_part = part_bf[neighbor_bf];
                if (p != neighbor_part) {
                    idx_t *node_halobfid_size = (idx_t *)PyArray_GETPTR2(node_halobfid, local_index, node_halobfid_dims[1] - 1);
                    *(idx_t *)PyArray_GETPTR2(node_halobfid, local_index, *node_halobfid_size) = map_bf_recv[neighbor_bf];
                    *node_halobfid_size += 1;
                }
            }

        }

        //# PhyFaces, PhyFacesName
        print_instant("PhyFaces, PhyFacesName\n");
        for (auto iter = map_phy_faces.begin(); iter != map_phy_faces.end(); ++iter) {
            const idx_t k = iter->first;
            const idx_t local_index = iter->second;

            l_phy_faces_name_data[local_index] = *(idx_t *)PyArray_GETPTR1(phy_faces_name, k);
            l_phy_faces_loctoglob_data[local_index] = k; //loctoglob

            const idx_t size = *(idx_t *)PyArray_GETPTR2(phy_faces, k, phyfaces_dim[1] - 1);
            for (idx_t i = 0; i < size; i++) {
                const idx_t g_node = *(idx_t *)PyArray_GETPTR2(phy_faces, k, i);
                l_phy_faces_data[local_index * (max_phy_face_nodeid + 1) + i] = map_nodes[g_node];
            }
            //size
            l_phy_faces_data[local_index * (max_phy_face_nodeid + 1) + max_phy_face_nodeid] = size;
        }



        // # Halo neighsub, HalosExt, shared_bf_recv, l_bf_recv_part_size_data, l_bf_cellid_data
        print_instant("Halo neighsub, Halos\n");

        idx_t counter = 0;
        idx_t bf_cellid_counter = 0;
        for (int iter : set_halo_neighsub) {
            l_halo_neighsub_data[counter] = iter;
            counter++;
        }

        counter = 0;
        idx_t tmp[3] = {0, -1, 0}; // counter, part, size
        for (auto &item: vec_shared_bf_recv) {
            l_shared_bf_recv_data[counter++] = item;


            const idx_t part = part_bf[item];
            if (p == part) {
                const idx_t cell_id = *(idx_t *)PyArray_GETPTR2(bf_cellid, item, 0);
                const idx_t face_index = *(idx_t *)PyArray_GETPTR2(bf_cellid, item, 1);
                l_bf_cellid_data[bf_cellid_counter * 2 + 0] = map_cells[cell_id];
                l_bf_cellid_data[bf_cellid_counter * 2 + 1] = face_index;
                bf_cellid_counter++;
            }
            if (tmp[1] != part) {
                if (tmp[1] != -1) {
                    l_bf_recv_part_size_data[tmp[0]++] = tmp[1];
                    l_bf_recv_part_size_data[tmp[0]++] = tmp[2];
                }
                tmp[1] = part;
                tmp[2] = 0;
            }
            tmp[2]++;


        }
        if (tmp[1] != -1) {
            l_bf_recv_part_size_data[tmp[0]++] = tmp[1];
            l_bf_recv_part_size_data[tmp[0]++] = tmp[2];
        }


        for (auto & map_halo : map_halos) {
            const idx_t k = map_halo.first;
            const idx_t local_index = map_halo.second;


            const idx_t size = *(idx_t *)PyArray_GETPTR2(cells, k, cells_dim[1] - 1);
            const idx_t data_size = (max_cell_nodeid + 2);
            const idx_t index = local_index * data_size;

            l_halo_halosext_data[index] = k;
            for (idx_t i = 0; i < size; i++) {
                const idx_t g_node = *(idx_t *)PyArray_GETPTR2(cells, k, i);
                l_halo_halosext_data[index + i + 1] = g_node;
            }
            l_halo_halosext_data[index + data_size - 1] = size;
        }

        print_instant("End Halo neighsub, Halos\n");


    }

    for (idx_t p = 0; p < nb_parts; p++) {
        idx_t tmp[3] = {-1, -1, -1};
        for (idx_t &item : local_domains[p].vec_shared_bf_recv) {
            const idx_t item_part = part_bf[item];
            if (item_part != p) {
                auto &vec = local_domains[item_part].vec_shared_bf_send;
                if (tmp[0] != item_part) {
                    if (tmp[0] != -1) {
                        local_domains[tmp[0]].vec_shared_bf_send[tmp[2]] = tmp[1];
                    }
                    vec.push_back(p);
                    vec.push_back(-1);
                    tmp[0] = item_part;
                    tmp[1] = 0;
                    tmp[2] = (idx_t)vec.size() - 1;
                }
                const idx_t val = local_domains[item_part].map_bf_recv[item];
                vec.push_back(val);
                tmp[1]++;
            }
        }
        if (tmp[0] != -1) {
            local_domains[tmp[0]].vec_shared_bf_send[tmp[2]] = tmp[1];
        }
    }

    for (idx_t p = 0; p < nb_parts; p++) {
        if (local_domains[p].create_shared_bf_send() == -1) {
            free(part_vert);
            free(part_bf);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate NumPy array");
            return -1;
        }
    }



    print_instant("End function\n");
    free(part_vert);
    free(part_bf);
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
    PyObject *node_bfid_obj = nullptr;
    PyObject *bf_cellid_obj = nullptr;
    PyObject *cells_obj = nullptr;
    PyObject *cell_cellfid_obj = nullptr; //only for max_cell_halofid
    PyObject *cell_cellnid_obj = nullptr; //only for max_cell_halonid
    PyObject *cells_type_obj = nullptr;
    PyObject *nodes_obj = nullptr;
    PyObject *phy_faces_obj = nullptr;
    PyObject *phy_faces_name_obj = nullptr;
    idx_t nb_part = 0;

    print_instant("Parse Input \n");
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOi", &graph_obj, &node_cellid_obj, &node_bfid_obj, &bf_cellid_obj, &cells_obj, &cell_cellfid_obj, &cell_cellnid_obj, &cells_type_obj, &nodes_obj, &phy_faces_obj, &phy_faces_name_obj, &nb_part))
        return nullptr;
    if (nb_part < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return nullptr;
    }

    /*
    *Use NPY_ARRAY_IN_ARRAY when you:
        Only read the data.
        Need it aligned and C-contiguous.
        Want NumPy to copy if necessary and handle the details for you.
     */
    PyArrayObject *graph = (PyArrayObject *)PyArray_FROM_OTF(graph_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_cellid = (PyArrayObject *)PyArray_FROM_OTF(node_cellid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_bfid = (PyArrayObject *)PyArray_FROM_OTF(node_bfid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *bf_cellid = (PyArrayObject *)PyArray_FROM_OTF(bf_cellid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells = (PyArrayObject *)PyArray_FROM_OTF(cells_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cell_cellfid = (PyArrayObject *)PyArray_FROM_OTF(cell_cellfid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cell_cellnid = (PyArrayObject *)PyArray_FROM_OTF(cell_cellnid_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cells_type = (PyArrayObject *)PyArray_FROM_OTF(cells_type_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nodes = (PyArrayObject *)PyArray_FROM_OTF(nodes_obj, float_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_obj, int_type, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *phy_faces_name = (PyArrayObject *)PyArray_FROM_OTF(phy_faces_name_obj, int_type, NPY_ARRAY_IN_ARRAY);

    if (!graph || !node_cellid || !node_bfid || !bf_cellid || !cells || !cell_cellfid || !cell_cellnid || !cells_type || !nodes || !phy_faces || !phy_faces_name) {
        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(node_bfid);
        Py_XDECREF(bf_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cell_cellfid);
        Py_XDECREF(cell_cellnid);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        return nullptr;
    }
    if (
        PyArray_NDIM(graph) != 2 || PyArray_TYPE(graph) != int_type || !PyArray_ISCONTIGUOUS(graph) ||
        PyArray_NDIM(node_cellid) != 2 || PyArray_TYPE(node_cellid) != int_type || !PyArray_ISCONTIGUOUS(node_cellid) ||
        PyArray_NDIM(node_bfid) != 2 || PyArray_TYPE(node_bfid) != int_type || !PyArray_ISCONTIGUOUS(node_bfid) ||
        PyArray_NDIM(bf_cellid) != 2 || PyArray_TYPE(bf_cellid) != int_type || !PyArray_ISCONTIGUOUS(bf_cellid) ||
        PyArray_NDIM(cells) != 2 || PyArray_TYPE(cells) != int_type || !PyArray_ISCONTIGUOUS(cells) ||
        PyArray_NDIM(cell_cellfid) != 2 || PyArray_TYPE(cell_cellfid) != int_type || !PyArray_ISCONTIGUOUS(cell_cellfid) ||
        PyArray_NDIM(cell_cellnid) != 2 || PyArray_TYPE(cell_cellnid) != int_type || !PyArray_ISCONTIGUOUS(cell_cellnid) ||
        PyArray_NDIM(cells_type) != 1 || PyArray_TYPE(cells_type) != int_type || !PyArray_ISCONTIGUOUS(cells_type) ||
        PyArray_NDIM(nodes) != 2 || PyArray_TYPE(nodes) != float_type || !PyArray_ISCONTIGUOUS(nodes) ||
        PyArray_NDIM(phy_faces) != 2 || PyArray_TYPE(phy_faces) != int_type || !PyArray_ISCONTIGUOUS(phy_faces) ||
        PyArray_NDIM(phy_faces_name) != 1 || PyArray_TYPE(phy_faces_name) != int_type || !PyArray_ISCONTIGUOUS(phy_faces_name)
        ) {
        PyErr_SetString(PyExc_TypeError, "Input Data Type Error");
        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(node_bfid);
        Py_XDECREF(bf_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cell_cellfid);
        Py_XDECREF(cell_cellnid);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        return nullptr;
    }


    print_instant("Start Creating SubDomains \n");
    LocalDomainStruct *local_domains = new (std::nothrow) LocalDomainStruct[nb_part];
    if (!local_domains ||
        create_sub_domains(graph, node_cellid, node_bfid, bf_cellid, cells, cell_cellfid, cell_cellnid, cells_type, nodes, phy_faces, phy_faces_name, nb_part, local_domains) == -1
        ) {

        Py_XDECREF(graph);
        Py_XDECREF(node_cellid);
        Py_XDECREF(node_bfid);
        Py_XDECREF(bf_cellid);
        Py_XDECREF(cells);
        Py_XDECREF(cell_cellfid);
        Py_XDECREF(cell_cellnid);
        Py_XDECREF(cells_type);
        Py_XDECREF(nodes);
        Py_XDECREF(phy_faces);
        Py_XDECREF(phy_faces_name);
        if (!local_domains)
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate local domain struct");
        delete[] local_domains;
        return nullptr;
    }
    // No need of those
    Py_XDECREF(graph);
    Py_XDECREF(node_cellid);
    Py_XDECREF(node_bfid);
    Py_XDECREF(bf_cellid);
    Py_XDECREF(cells);
    Py_XDECREF(cell_cellfid);
    Py_XDECREF(cell_cellnid);
    Py_XDECREF(cells_type);
    Py_XDECREF(nodes);
    Py_XDECREF(phy_faces);
    Py_XDECREF(phy_faces_name);

    PyObject *py_list_result = PyList_New(nb_part);
    if (!py_list_result) {
        delete[] local_domains;
        return nullptr;
    }

    print_instant("Return List \n");
    for (int i = 0; i < nb_part; i++) {
        if (local_domains[i].create_tuple() == -1) {
            Py_XDECREF(py_list_result);
            delete[] local_domains;
            return nullptr;
        }
    }


    for (int i = 0; i < nb_part; i++) {
        PyList_SET_ITEM(py_list_result, i, local_domains[i].tuple_res);
        //the ownership transferred to the list.
        local_domains[i].tuple_res = nullptr;
    }

    delete[] local_domains;
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

