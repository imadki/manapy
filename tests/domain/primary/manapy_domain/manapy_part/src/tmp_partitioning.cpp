#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"


struct TmpLocalDomainStruct {
    TmpLocalDomainStruct(const TmpLocalDomainStruct&) = delete;                   // delete copy constructor
    TmpLocalDomainStruct& operator=(const TmpLocalDomainStruct&) = delete;        // delete copy assignment
    TmpLocalDomainStruct()= default;


    PyArray<double, 2> *nodes = nullptr; // float64[:, :] [[node x, y, z]]
    PyArray<int32_t, 2> *cells = nullptr; // int32[:, :] [[cells nodes]]
    PyArray<int8_t, 1> *cells_type = nullptr; // int8[:] [cell type]
    PyArray<int32_t, 2> *phy_faces = nullptr; // int32[:, :] [[physical face nodes]]
    PyArray<int32_t, 1> *phy_faces_name = nullptr; // int32[:] [physical face name]
    PyArray<int32_t, 1> *cell_loctoglob = nullptr; // int32[:] [cell global index]
    PyArray<int32_t, 1> *node_loctoglob = nullptr; // int32[:] [node global index]
    PyArray<int32_t, 1> *node_oldname = nullptr; // int32[:] [node old name, ...]
    PyArray<int32_t, 2> *halo_neighsub = nullptr; // int32[:, :] [[NeighborP1, NeighborP2, ...], [NbHalosIntConnectedToP1, ...]]
    PyArray<int32_t, 1> *node_halos = nullptr; // int32[:] [node1, number of halos, halocell index in halo_halosext, node2, number of halos, ....] shape=(2*nb_nodes + nb_halos)
    PyArray<int32_t, 2> *node_halophyid = nullptr; // int32[:, :] [[index0 point to halo_halobf, index1 ..., size]] shape=(nb_nodes, max_node_halobf + 1)
    PyArray<int32_t, 2> *halo_halosext = nullptr; // int32[:, :] [[global index of halocell, global index of cell nodes, size]] shape=(nb_halos, max_cell_nodeid + 2)
    PyArray<int32_t, 1> *halo_halosint = nullptr; // int32[:] [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    PyArray<int32_t, 1> *phyid_recv = nullptr; // int32[:] [boundary faces global index, ...] description="store physical faces of this partition by its local index and for the other partitions by global index, all other tables that will use boundary faces must point to this table"
    PyArray<int32_t, 1> *phyid_recv_part_size = nullptr; // int32[:] [boundary faces partId, size]
    PyArray<int32_t, 1> *phyid_send = nullptr;  // int32[:] self.phyid_send = np.zeros(1, dtype=np.int32) # [recv_part_index, size, size indices point to phyid_recv, ...] description="used when this part need to send its boundary faces to recv_part"
    PyObject *tuple_res = nullptr;

    // Scalars
    int max_cell_nodeid = 0;
    int max_cell_faceid = 0;
    int max_face_nodeid = 0;
    int max_node_haloid = 0;
    int max_cell_halonid = 0;



    // Temporary members used to generate the above tables and scalars
    std::map<int, std::vector<int> > map_int_halos;
    std::vector<int32_t> b_nodes;
    int max_node_halophyid = 0;
    int max_phy_face_nodeid = 0;
    int nb_node_halos = 0;
    //
    std::map<int, int> map_phy_faces;
    std::set<int> set_phyids;
    std::set<int> set_halo_phyid_neighsub;
    std::vector<int32_t> vec_phyids;
    std::map<int32_t, int32_t> map_phyids;



    ~TmpLocalDomainStruct();


public:
    void _create_tables(PyArray<double, 2> *nodes, std::vector<int32_t> &part_phyid);

    void create_tuple();

private:

    void    free_tables();

};

TmpLocalDomainStruct::~TmpLocalDomainStruct() {
    this->free_tables();
    Py_XDECREF(this->tuple_res);
    this->tuple_res = nullptr;
}

void TmpLocalDomainStruct::create_tuple() {
    PyObject *tuple = Py_BuildValue("(OOOOOOOOOOOOOOOOiiiii)",
        this->nodes->ref_holder,
        this->cells->ref_holder,
        this->cells_type->ref_holder,
        this->phy_faces->ref_holder,
        this->phy_faces_name->ref_holder,
        this->cell_loctoglob->ref_holder,
        this->node_loctoglob->ref_holder,
        this->node_oldname->ref_holder,
        this->halo_neighsub->ref_holder,
        this->node_halos->ref_holder,
        this->node_halophyid->ref_holder,
        this->phyid_recv->ref_holder,
        this->phyid_recv_part_size->ref_holder,
        this->phyid_send->ref_holder,
        this->halo_halosext->ref_holder,
        this->halo_halosint->ref_holder,
        this->max_cell_nodeid,
        this->max_cell_faceid,
        this->max_face_nodeid,
        this->max_node_haloid,
        this->max_cell_halonid);
    if (!tuple) {
        throw std::bad_alloc();
    }

    // tuple hold references now
    this->free_tables();
    this->tuple_res = tuple;
}


// private
void TmpLocalDomainStruct::free_tables() {
    // all these tables are created using this->_create_tables
    delete this->nodes; this->nodes = nullptr;
    delete this->cells; this->cells = nullptr;
    delete this->cells_type; this->cells_type = nullptr;
    delete this->phy_faces; this->phy_faces = nullptr;
    delete this->phy_faces_name; this->phy_faces_name = nullptr;
    delete this->cell_loctoglob; this->cell_loctoglob = nullptr;
    delete this->node_loctoglob; this->node_loctoglob = nullptr;
    delete this->node_oldname; this->node_oldname = nullptr;
    delete this->halo_neighsub; this->halo_neighsub = nullptr;
    delete this->node_halos; this->node_halos = nullptr;
    delete this->node_halophyid; this->node_halophyid = nullptr;
    delete this->phyid_recv; this->phyid_recv = nullptr;
    delete this->phyid_recv_part_size; this->phyid_recv_part_size = nullptr;
    delete this->phyid_send; this->phyid_send = nullptr;
    delete this->halo_halosext; this->halo_halosext = nullptr;
    delete this->halo_halosint; this->halo_halosint = nullptr;
}

static int32_t binary_search(const PyArray<int32_t, 1> &arr, const int32_t item) {
    const int32_t size = arr.last();
    int32_t left = 0;
    int32_t right = size - 1;
    while (left <= right) {
        const int32_t mid = (left + right) / 2;
        const int32_t mid_val = arr.get(mid);
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

static void intersect_arr(PyArray<int32_t, 2> *arr, PyArray<int32_t, 1> *indices, const int32_t size, std::vector<int32_t> &intersect_arr) {
    int32_t counter = 0;

    intersect_arr[0] = -1;
    intersect_arr[1] = -1;

    auto arr1 = arr->sub_array(indices->get(0));
    for (int32_t i = 0; i < arr1.last(); i++) {
        intersect_arr[counter] = arr1.get(i);
        for (int32_t j = 1; j < size; j++) {
            auto arr2 = arr->sub_array(indices->get(j));
            if (binary_search(arr2, arr1.get(i)) == -1){
                intersect_arr[counter] = -1;
                break;
            }
        }
        if (intersect_arr[counter] != -1)
            counter++;
        if (counter >= 2)
            break;
    }
}


static std::vector<int32_t> _get_max_info(const int32_t cell_type) {
    if (cell_type == CELL_TYPE::Triangle) {
        return {3, 2, 3};
    }
    else if (cell_type == CELL_TYPE::Quad) {
        return {4, 2, 4};
    }
    else if (cell_type == CELL_TYPE::Tetra) {
        return {4, 3, 4};
    }
    else if (cell_type == CELL_TYPE::Hexahedron) {
        return {6, 4, 8};
    }
    else if (cell_type == CELL_TYPE::Pyramid) {
        return {5, 4, 5};
    }
    return {0, 0, 0};
}

// #################################################################
// 1. _create_sub_domains
// #################################################################

static void create_halos(
    TmpLocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *cells,
    PyArray<int32_t, 2> *node_cellid,
    const int32_t p) {

    auto l_node_loctoglob = ld[p].node_loctoglob;
    auto l_map_int_halos = ld[p].map_int_halos;
    auto &l_b_nodes = ld[p].b_nodes;
    const int32_t l_max_cell_nodeid = ld[p].max_cell_nodeid;
    const int32_t l_nb_node_halos = ld[p].nb_node_halos;

    // #########################################################
    // max_node_haloid, l_node_halos
    // #########################################################
    // TODO check if nb_node_halos is accurate
    ld[p].node_halos = new PyArray<int32_t, 1>(make_npy_dims(l_nb_node_halos));
    const auto l_node_halos = ld[p].node_halos;

    std::map<int32_t, int32_t> map_halos;
    int32_t halos_counter = 0;

    ld[p].max_node_haloid = -1;
    int32_t &max_node_haloid = ld[p].max_node_haloid;

    for (const int32_t local_node_id : l_b_nodes) {
        const int32_t g_index = l_node_loctoglob->get(local_node_id);
        auto sub_node_cellid = node_cellid->sub_array(g_index);

        int32_t node_counter = -1;
        for (int32_t i = 0; i < sub_node_cellid.last(); i++) {
            const int32_t neighbor_cell = sub_node_cellid.get(i);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);
            if (neighbor_part != p) {

                // get ext halos
                if (map_halos.find(neighbor_cell) == map_halos.end()) {
                    map_halos[neighbor_cell] = (int32_t)map_halos.size();
                }

                if (node_counter == -1) {
                    // set [nodeid, size=0] and node_counter index
                    l_node_halos->get(halos_counter) = local_node_id;
                    l_node_halos->get(halos_counter + 1) = 0;
                    node_counter = halos_counter + 1;
                    halos_counter += 2;
                }
                // append node_halos
                l_node_halos->get(halos_counter) = map_halos[neighbor_cell];
                halos_counter += 1;
                l_node_halos->get(node_counter) += 1; // [nodeid, size++]

            }
        }

        if (node_counter != -1) {
            max_node_haloid = std::max(l_node_halos->get(node_counter), max_node_haloid);
        }
    }


    // #########################################################
    // l_halo_halosext
    // #########################################################

    ld[p].halo_halosext = new PyArray<int32_t, 2>(make_npy_dims(map_halos.size(), l_max_cell_nodeid + 2));
    const auto *l_halo_halosext = ld[p].halo_halosext;

    for (const auto &item: map_halos) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        auto sub_l_halo_halosext = l_halo_halosext->sub_array(l_id);
        sub_l_halo_halosext.get(0) = g_id;
        auto sub_cells = cells->sub_array(g_id);
        for (int32_t j = 0; j < sub_cells.last(); j++) {
            const int32_t nodeid = sub_cells.get(j);
            sub_l_halo_halosext.get(j + 1) = nodeid;
        }
        sub_l_halo_halosext.last() = sub_cells.last() + 1;
    }

    // #########################################################
    // l_halo_neighsub, l_halo_halosint
    // #########################################################

    int32_t nb_halos_int = 0;
    for (const auto &item : l_map_int_halos) {
        nb_halos_int += (int32_t)item.second.size();
    }

    ld[p].halo_neighsub = new PyArray<int32_t, 2>(make_npy_dims(2, l_map_int_halos.size()));
    ld[p].halo_halosint = new PyArray<int32_t, 1>(make_npy_dims(nb_halos_int));
    const auto *l_halo_neighsub = ld[p].halo_neighsub;
    const auto *l_halo_halosint = ld[p].halo_halosint;

    int32_t neighsub_counter = 0;
    int32_t halosint_counter = 0;
    for (const auto &item : l_map_int_halos) {
        const int32_t partition = item.first;
        const auto &set = item.second;

        l_halo_neighsub->sub_array(0).get(neighsub_counter) = partition;
        l_halo_neighsub->sub_array(1).get(neighsub_counter) = (int32_t)set.size();
        neighsub_counter += 1;
        for (const auto interior_cell: set) {
            l_halo_halosint->get(halosint_counter) = interior_cell;
            halosint_counter += 1;
        }
    }
}

void    phy_p(
    TmpLocalDomainStruct *ld,
    const int32_t p,
    PyArray<int32_t, 2> *node_phyid,
    PyArray<int32_t, 1> *phy_faces_name,
    PyArray<int32_t, 2> *phy_faces,
    const std::vector<int32_t> &part_phyid,
    const std::vector<int32_t> &vec_node_oldname,
    const std::vector<std::map<int32_t, int32_t> > &vec_map_nodes
    ) {

    auto &l_set_phyids = ld[p].set_phyids;
    auto &l_map_phy_faces = ld[p].map_phy_faces;
    const int32_t nb_halo_phyid_neighsub = (int32_t)ld[p].set_halo_phyid_neighsub.size();
    const int32_t max_node_halophyid = ld[p].max_node_halophyid;
    const int32_t max_phy_face_nodeid = ld[p].max_phy_face_nodeid;
    auto l_node_loctoglob = ld[p].node_loctoglob;

    const int32_t l_nb_nodes = (int32_t)l_node_loctoglob->shape[0];

    // #########################################################
    // phyid_recv => [phyid of p_a, ..., phyid of p_b, ...]
    // phyid_recv_part_size => [partition Id, size, ...]
    // #########################################################

    ld[p].vec_phyids = std::vector<int32_t>(l_set_phyids.size());
    ld[p].map_phyids = std::map<int32_t, int32_t>();
    auto &vec_phyids = ld[p].vec_phyids;
    auto &map_phyids = ld[p].map_phyids;

    int32_t vec_phyids_counter = 0;
    for (const auto item : l_set_phyids) {
        vec_phyids[vec_phyids_counter] = item;
        vec_phyids_counter += 1;
    }
    // Sort vec_phyids by comparing part_phyid[phyid]
    std::sort(vec_phyids.begin(), vec_phyids.end(),[&part_phyid](const int a, const int b) {
        return part_phyid[a] < part_phyid[b];
    });

    for (int32_t i = 0; i < vec_phyids.size(); i++) {
        const int32_t item = vec_phyids[i];
        map_phyids[item] = i;
    }

    // Create the tables
    ld[p].phyid_recv = new PyArray<int32_t, 1>(make_npy_dims(vec_phyids.size()));
    ld[p].phyid_recv_part_size = new PyArray<int32_t, 1>(make_npy_dims(nb_halo_phyid_neighsub * 2 + 2));
    auto l_phyid_recv = ld[p].phyid_recv;
    auto l_phyid_recv_part_size = ld[p].phyid_recv_part_size;

    int32_t old_part = -1;
    int32_t counter = 0;
    bool p_has_halo_phyid = false;
    for (int32_t i = 0; i < vec_phyids.size(); i++) {
        const int32_t g_id = vec_phyids[i];
        const int32_t part = part_phyid[g_id];
        l_phyid_recv->get(i) = g_id;
        if (p == part) {
            p_has_halo_phyid = true;
            l_phyid_recv->get(i) = l_map_phy_faces.at(g_id); // transform phyid to local for p == part
        }
        if (old_part != part) {
            l_phyid_recv_part_size->get(counter) = part;
            l_phyid_recv_part_size->get(counter + 1) = 0;
            old_part = part;
            counter += 2;
        }
        l_phyid_recv_part_size->get(counter - 1) += 1;
    }
    if (!p_has_halo_phyid) {
        l_phyid_recv_part_size->get(counter) = p;
        l_phyid_recv_part_size->get(counter + 1) = 0;
    }



    // #########################################################
    // l_node_oldname, l_node_halophyid
    // #########################################################

    ld[p].node_oldname = new PyArray<int32_t, 1>(make_npy_dims(l_nb_nodes));
    ld[p].node_halophyid = new PyArray<int32_t, 2>(make_npy_dims(l_nb_nodes, max_node_halophyid + 1));
    auto l_node_oldname = ld[p].node_oldname;
    auto l_node_halophyid = ld[p].node_halophyid;

    // TODO max_node_haloid already calculated
    for (int32_t l_id = 0; l_id < l_nb_nodes; l_id++) {
        const int32_t g_id = l_node_loctoglob->get(l_id);

        l_node_oldname->get(l_id) = vec_node_oldname[g_id];

        // l_node_halophyid
        const auto sub_node_phyid = node_phyid->sub_array(g_id);
        auto sub_l_node_halophyid = l_node_halophyid->sub_array(l_id);
        int32_t node_halophyid_counter = 0;
        for (int32_t j = 0; j < sub_node_phyid.last(); j++) {
            const int32_t neighbor_phyid = sub_node_phyid.get(j);
            const int32_t neighbor_part = part_phyid[neighbor_phyid];
            if (p != neighbor_part) {
                sub_l_node_halophyid.get(node_halophyid_counter) = map_phyids.at(neighbor_phyid);
                node_halophyid_counter++;
            }
        }
        sub_l_node_halophyid.last() = node_halophyid_counter;
    }


    // #########################################################
    // l_phy_faces, l_phy_faces_name
    // #########################################################

    ld[p].phy_faces = new PyArray<int32_t, 2>(make_npy_dims(l_map_phy_faces.size(), max_phy_face_nodeid + 1));
    ld[p].phy_faces_name = new PyArray<int32_t, 1>(make_npy_dims(l_map_phy_faces.size()));
    auto l_phy_faces = ld[p].phy_faces;
    auto l_phy_faces_name = ld[p].phy_faces_name;

    for (const auto &item : l_map_phy_faces) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        l_phy_faces_name->get(l_id) = phy_faces_name->get(g_id);

        const auto the_phy_faces = phy_faces->sub_array(g_id);
        auto the_l_phy_faces = l_phy_faces->sub_array(l_id);
        for (int32_t j = 0; j < the_phy_faces.last(); j++) {
            const int32_t nodeid = the_phy_faces.get(j);
            the_l_phy_faces.get(j) = vec_map_nodes[nodeid].at(p);
        }
        the_l_phy_faces.last() = the_phy_faces.last();
    }
}

static void _create_phyid_send(TmpLocalDomainStruct *ld, const int32_t nb_parts) {
    // #########################################################
    // phyid_send => [partition_id, size, indices point to phyid_recv_part_size, ...]
    // #########################################################
    std::vector<std::vector<int32_t> > vec_list_phyid_send(nb_parts);

    for (int32_t p = 0; p < nb_parts; p++) {
        auto &vec_phyids = ld[p].vec_phyids;
        auto phyid_recv_part_size = ld[p].phyid_recv_part_size;
        int32_t counter = 0;

        for (int32_t i = 0; i < phyid_recv_part_size->shape[0]; i += 2) {
            const int32_t part = phyid_recv_part_size->get(i);
            const int32_t size = phyid_recv_part_size->get(i + 1);
            if (part != p) {
                auto &list_phyid_send = vec_list_phyid_send[part];
                auto &map_phyids = ld[part].map_phyids;
                list_phyid_send.push_back(p);
                list_phyid_send.push_back(size);
                for (int32_t j = 0; j < size; j++) {
                    const int32_t phy_id = vec_phyids[counter + j];
                    const int32_t index = map_phyids[phy_id];
                    list_phyid_send.push_back(index);
                }
            }
            counter += size;
        }
    }

    for (int32_t p = 0; p < nb_parts; p++) {
        auto &list_phyid_send = vec_list_phyid_send[p];
        ld[p].phyid_send = new PyArray<int32_t, 1>(make_npy_dims(list_phyid_send.size()));
        for (int32_t i = 0; i < list_phyid_send.size(); i++) {
            ld[p].phyid_send->get(i) = list_phyid_send[i];
        }
    }
}




void    devid_nodes(
    TmpLocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<double, 2> *nodes,
    std::vector<bool> &node_is_boundary,
    std::vector<std::map<int32_t, int32_t> > &vec_map_nodes,
    const int32_t nb_parts
) {
    std::vector<int32_t> parts; // temporarily vector to store node neighboring parts
    std::vector<int32_t> parts_counter(nb_parts, 0); // for a fixed node `i` determine the number of neighbors for each part
    std::vector<int32_t> local_nodes_counter(nb_parts, 0); // determine the number of nodes for each parts
    std::vector<int32_t> local_boundary_nodes_counter(nb_parts, 0); // determine the number of boundary nodes for each parts
    parts.reserve(100);

    for (int32_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        for (int32_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t neighbor_cell = sub_node_cellid.get(j);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);

            if (parts_counter[neighbor_part] == 0) {
                parts.push_back(neighbor_part);
            }
            parts_counter[neighbor_part]++;
        }
        for (int32_t j = 0; j < parts.size(); j++) {
            const int32_t part = parts[j];

            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;

            if (parts.size() > 1) {
                // this is a boundary node
                local_boundary_nodes_counter[part]++;
            }

            // reset parts_counter
            parts[j] = 0;
            parts_counter[part] = 0;
        }

        // reset the size of the vector
        parts.clear();
    }

    // #########################################################
    // creating: local_nodes, local_node_loctoglob, local_b_nodes, nb_node_halos
    // #########################################################


    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_nodes = local_nodes_counter[i];
        const int32_t nb_b_nodes = local_boundary_nodes_counter[i];

        ld[i].nodes = new PyArray<double, 2>(make_npy_dims(nb_nodes, nodes->shape[1]));
        ld[i].node_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(nb_nodes));
        ld[i].b_nodes.resize(nb_b_nodes);
    }


    std::fill(parts_counter.begin(), parts_counter.end(), 0);
    std::fill(local_nodes_counter.begin(), local_nodes_counter.end(), 0);
    std::fill(local_boundary_nodes_counter.begin(), local_boundary_nodes_counter.end(), 0);
    for (int32_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        const auto sub_nodes = nodes->sub_array(i);

        for (int32_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t neighbor_cell = sub_node_cellid.get(j);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);
            const int32_t local_nodeid = local_nodes_counter[neighbor_part];

            if (parts_counter[neighbor_part] == 0) {
                parts.push_back(neighbor_part);

                //* assign node_loctoglob
                ld[neighbor_part].node_loctoglob->get(local_nodeid) = i;
                vec_map_nodes[i][neighbor_part] = local_nodeid;
            }

            //increment parts_counter for a neighbor_part
            parts_counter[neighbor_part]++;


            //* assign local_nodes
            const auto sub_l_nodes = ld[neighbor_part].nodes->sub_array(local_nodeid);
            for (int32_t k = 0; k < nodes->shape[1]; k++) {
                sub_l_nodes.get(k) = sub_nodes.get(k);
            }
        }

        for (int32_t j = 0; j < parts.size(); j++) {
            const int32_t part = parts[j];
            if (parts.size() > 1) {
                // this is a boundary node

                const int32_t local_nodeid = local_nodes_counter[part];
                int32_t &counter = local_boundary_nodes_counter[part];

                //* assign b_nodes
                ld[part].b_nodes[counter] = local_nodeid;
                counter++;

                //* assign local_nodes_is_boundary
                node_is_boundary[i] = true;

                //* assign nb_node_halos (increment only if the node is halo parts.size() > 1)
                const int32_t nb_part_halos = sub_node_cellid.last() - parts_counter[part];
                ld[part].nb_node_halos += 2; // increment for node, increment for size [nodeid, size]
                ld[part].nb_node_halos += nb_part_halos; // increment for node halos [nodeid, size, halos...]
            }

            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;




            // reset parts_counter
            parts[j] = 0;
            parts_counter[part] = 0;
        }

        // reset the size of the vector
        parts.clear();
    }
}

void    devid_phy(
    TmpLocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    std::vector<int32_t> &vec_node_oldname,
    std::vector<int32_t> &part_phyid
) {

    // #########################################################
    // Create Physical faces And node old name, boundary_cells
    // max_phy_face_nodeid, map_phy_faces, part_phyid, vec_node_oldname
    // #########################################################

    print_instant("\t2.1. Create Physical faces And node old name, boundary_cells\n");

    std::vector<int32_t> intersect_cell(2);
    std::vector<int32_t> boundary_cells(phy_faces->shape[0]); //cells that has at least one physical face attached to it

    int total_nb_phyfaces = 0;
    for (idx_t i = 0; i < phy_faces->shape[0]; i++) {
        auto phy_face = phy_faces->sub_array(i);
        const idx_t name = phy_faces_name->get(i);
        const idx_t size = phy_face.last();
        intersect_arr(node_cellid, &phy_face, size, intersect_cell);
        if (intersect_cell[0] != -1) {
            const int32_t p = part_vert->get(intersect_cell[0]);
            ld[p].max_phy_face_nodeid = std::max(size, ld[p].max_phy_face_nodeid);

            auto &tmp_map = ld[p].map_phy_faces;
            tmp_map[i] = (int32_t)tmp_map.size();
            boundary_cells[total_nb_phyfaces] = intersect_cell[0];
            total_nb_phyfaces++;
        }
        for (int32_t j = 0; j < size; j++) {
            const int32_t nodeid = phy_face.get(j);
            if (vec_node_oldname[nodeid] == 0 or vec_node_oldname[nodeid] > name)
                vec_node_oldname[nodeid] = name;
        }
    }

    // #########################################################
    // Create part_phyid
    // #########################################################
    print_instant("\t2.2. Create part_phyid\n");

    for (int32_t phyid = 0; phyid < boundary_cells.size(); phyid++) {
        const int32_t cell_id = boundary_cells[phyid];
        part_phyid[phyid] = part_vert->get(cell_id);
    }
}

// TODO try vector of LocalDomain Vs LocalDomain of vect

void    devid_cells(
TmpLocalDomainStruct *ld,
PyArray<int32_t, 1> *part_vert,
PyArray<int32_t, 2> *node_cellid,
PyArray<int32_t, 2> *cells,
PyArray<int8_t, 1> *cells_type,
PyArray<int32_t, 2> *node_phyid,
const std::vector<bool> &node_is_boundary,
const std::vector<std::map<int32_t, int32_t> > &vec_map_nodes,
const std::vector<int32_t> &part_phyid,
const int32_t nb_parts
) {
        // #########################################################
    // creating:  local_cells, local_cells_type, local_cell_loctoglob, local_max_cell_nodeid, local_max_cell_faceid, local_max_face_nodeid, local_max_cell_halonid, max_node_halophyid, set_phyid, set_halo_phyid_neighsub
    // #########################################################

    std::vector<int32_t> local_nb_cells(nb_parts, 0);
    std::vector<int32_t> i_visited(cells->shape[0], -1);

    for (int32_t i = 0; i < cells->shape[0]; i++) {
        const int32_t part = part_vert->get(i);
        const int8_t cell_type = cells_type->get(i);
        const auto max_info = _get_max_info(cell_type);

        ld[part].max_cell_faceid = std::max(max_info[0], ld[part].max_cell_faceid);
        ld[part].max_face_nodeid = std::max(max_info[1], ld[part].max_face_nodeid);
        ld[part].max_cell_nodeid = std::max(max_info[2], ld[part].max_cell_nodeid);
        local_nb_cells[part]++;
    }



    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_cells = local_nb_cells[i];
        const int32_t max_cell_nodeid = ld[i].max_cell_nodeid;

        ld[i].cells = new PyArray<int32_t, 2>(make_npy_dims(nb_cells, max_cell_nodeid + 1));
        ld[i].cells_type = new PyArray<int8_t, 1>(make_npy_dims(nb_cells));
        ld[i].cell_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(nb_cells));
    }

    std::fill(local_nb_cells.begin(), local_nb_cells.end(), 0);
    for (int32_t g_id = 0; g_id < cells->shape[0]; g_id++) {
        const int32_t part = part_vert->get(g_id);
        const int32_t l_id = local_nb_cells[part];
        const int8_t cell_type = cells_type->get(g_id);
        auto &map_int_halos = ld[part].map_int_halos;

        //* assign local_cells_type
        ld[part].cells_type->get(l_id) = cell_type;

        //* assign l_cell_loctoglob
        ld[part].cell_loctoglob->get(l_id) = g_id;

        //* assign local_cells
        const auto sub_cells = cells->sub_array(g_id);
        auto sub_l_cells = ld[part].cells->sub_array(l_id);
        int32_t nb_cell_halonid = 0;
        for (int32_t j = 0; j < sub_cells.last(); j++) {
            const int32_t nodeid = sub_cells.get(j);
            const int32_t local_nodeid = vec_map_nodes[nodeid].at(part);
            sub_l_cells.get(j) = local_nodeid;

            if (node_is_boundary[nodeid]) {
                auto sub_node_cellid = node_cellid->sub_array(nodeid);
                for (int32_t k = 0; k < sub_node_cellid.last(); k++) {
                    const int32_t neighbor_cell = sub_node_cellid.get(k);
                    const int32_t neighbor_part = part_vert->get(neighbor_cell);
                    if (neighbor_part != part and i_visited[neighbor_cell] != g_id) {
                        i_visited[neighbor_cell] = g_id;
                        nb_cell_halonid++;

                        // interior halos

                        if (map_int_halos.find(neighbor_part) == map_int_halos.end()) {
                            map_int_halos[neighbor_part] = std::vector<int32_t>();
                        }
                        auto &set_int_halos = map_int_halos[neighbor_part];
                        if (set_int_halos.empty() or set_int_halos.back() != g_id) {
                            set_int_halos.push_back(g_id);
                        }
                    }
                }
            }

            // max_node_halophyid, set_phyid, set_halo_phyid_neighsub
            int32_t nb_node_halophyid = 0;
            auto the_node_phyid = node_phyid->sub_array(nodeid);
            for (int32_t k = 0; k < the_node_phyid.last(); k++) {
                const int32_t phy_id = the_node_phyid.get(k);
                const int32_t phy_id_part = part_phyid[phy_id];
                if (ld[part].set_phyids.count(phy_id) == 0) {
                    ld[part].set_phyids.insert(phy_id);
                }
                if (part != phy_id_part) {
                    nb_node_halophyid += 1;
                    ld[part].set_halo_phyid_neighsub.insert(phy_id_part);
                }
            }
            ld[part].max_node_halophyid = std::max(nb_node_halophyid, ld[part].max_node_halophyid);
        }
        sub_l_cells.last() = sub_cells.last();

        //* assign local_max_cell_halonid
        ld[part].max_cell_halonid = std::max(ld[part].max_cell_halonid, nb_cell_halonid);


        local_nb_cells[part]++;
    }
}

static PyObject *get_result_as_py_list(TmpLocalDomainStruct *ld, const int32_t nb_parts) {
    PyObject *py_list_result = PyList_New(nb_parts);
    if (!py_list_result) {
        throw std::bad_alloc();
    }
    for (int i = 0; i < nb_parts; i++) {
        ld[i].create_tuple(); // create ld[i].tuple_res
        PyList_SET_ITEM(py_list_result, i, ld[i].tuple_res);

        // The ownership transferred to the list.
        ld[i].tuple_res = nullptr;
    }

    return py_list_result;
}

PyObject *devide(
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<double, 2> *nodes,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    PyArray<int32_t, 2> *node_phyid,
    const int32_t nb_parts
    ) {

    print_instant("Start devide\n");
    auto *ld = new(std::nothrow) TmpLocalDomainStruct[nb_parts];

    time_it(""); // start time
    std::vector<int32_t> vec_node_oldname(nodes->shape[0]);
    std::vector<int32_t> part_phyid(phy_faces->shape[0]);
    std::vector<bool> node_is_boundary(nodes->shape[0], false);
    std::vector<std::map<int32_t, int32_t> > vec_map_nodes(nodes->shape[0]); // for every g_node store local id of a certain partition

    devid_nodes(ld, part_vert, node_cellid, nodes, node_is_boundary, vec_map_nodes, nb_parts);
    devid_phy(ld, part_vert, node_cellid, phy_faces, phy_faces_name, vec_node_oldname, part_phyid);
    devid_cells(ld, part_vert, node_cellid, cells, cells_type, node_phyid, node_is_boundary, vec_map_nodes, part_phyid, nb_parts);
    time_it("devide");

    // #########################################################
    // creating:  local_cells, local_cells_type, local_cell_loctoglob, local_max_cell_nodeid, local_max_cell_faceid, local_max_face_nodeid, local_max_cell_halonid
    // #########################################################
    print_instant("Create Locals\n");
    for (int32_t p = 0; p < nb_parts; p++) {
        time_it("");
        create_halos(ld, part_vert, cells, node_cellid, p);
        phy_p(ld, p, node_phyid, phy_faces_name, phy_faces, part_phyid, vec_node_oldname, vec_map_nodes);
        time_it("create_halos");
    }
    _create_phyid_send(ld, nb_parts);

    PyObject *py_list_result = get_result_as_py_list(ld, nb_parts);

    return py_list_result;
}




