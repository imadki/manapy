#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"
#include "LocalDomainStruct.h"

// #################################################################
// 1. _create_sub_domains
// #################################################################

static int32_t _binary_search(const PyArray<int32_t, 1> &arr, const int32_t item) {
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

static void _intersect_arr(PyArray<int32_t, 2> *arr, PyArray<int32_t, 1> *indices, const int32_t size, std::vector<int32_t> &intersect_arr) {
    int32_t counter = 0;

    intersect_arr[0] = -1;
    intersect_arr[1] = -1;

    auto arr1 = arr->sub_array(indices->get(0));
    for (int32_t i = 0; i < arr1.last(); i++) {
        intersect_arr[counter] = arr1.get(i);
        for (int32_t j = 1; j < size; j++) {
            auto arr2 = arr->sub_array(indices->get(j));
            if (_binary_search(arr2, arr1.get(i)) == -1){
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


static void _create_sub_domains(
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<int32_t, 2> *node_phyid,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    LocalDomainStruct *local_domains,
    std::vector<int32_t> &i_visited,
    std::vector<int32_t> &vec_node_oldname,
    std::vector<int32_t> &intersect_cell,
    std::vector<int32_t> &boundary_cells,
    std::vector<int32_t> &part_phyid) {

    // #########################################################
    // Create Physical faces And node old name, boundary_cells
    // #########################################################
    print_instant("\t2.1. Create Physical faces And node old name, boundary_cells\n");

    int total_nb_phyfaces = 0;
    for (idx_t i = 0; i < phy_faces->shape[0]; i++) {
        auto phy_face = phy_faces->sub_array(i);
        const idx_t name = phy_faces_name->get(i);
        const idx_t size = phy_face.last();
        _intersect_arr(node_cellid, &phy_face, size, intersect_cell);
        if (intersect_cell[0] != -1) {
            const int32_t p = part_vert->get(intersect_cell[0]);
            local_domains[p].max_phy_face_nodeid = std::max(size, local_domains[p].max_phy_face_nodeid);
            local_domains[p].map_phy_faces[i] = (int32_t)local_domains[p].map_phy_faces.size();
            boundary_cells[total_nb_phyfaces] = intersect_cell[0];
            total_nb_phyfaces++;
        }
        for (int32_t j = 0; j < size; j++) {
            const int32_t nodeid = phy_face.get(j);
            if (vec_node_oldname[nodeid] == 0 or vec_node_oldname[nodeid] > name)
                vec_node_oldname[nodeid] = name;
        }
    }

    if (total_nb_phyfaces != phy_faces->shape[0]) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Warning: not all the physical faces match the domain faces !! %d "
            "where the number of physical faces is %ld",
            total_nb_phyfaces, phy_faces->shape[0]);
        throw std::runtime_error(msg);
    }

    // #########################################################
    // Create part_phyid
    // #########################################################
    print_instant("\t2.2. Create part_phyid\n");

    for (int32_t phyid = 0; phyid < boundary_cells.size(); phyid++) {
        const int32_t cell_id = boundary_cells[phyid];
        part_phyid[phyid] = part_vert->get(cell_id);
    }

    // #########################################################
    // map_cells, map_nodes, map_halos, map_halo_int, set_phyid, set_halo_phyid_neighsub
    // max_cell_faceid, max_face_nodeid, max_cell_nodeid, max_cell_halonid, nb_node_halos, max_node_halophyid
    // #########################################################
    print_instant("\t2.3. map_cells, map_nodes, map_halos, map_halo_int, set_phyid, set_halo_phyid_neighsub\n");

    for (int32_t i = 0; i < cells->shape[0]; i++) {
        const int32_t p = part_vert->get(i);
        const int8_t cell_type = cells_type->get(i);
        const auto max_info = _get_max_info(cell_type);
        local_domains[p].max_cell_faceid = std::max(max_info[0], local_domains[p].max_cell_faceid);
        local_domains[p].max_face_nodeid = std::max(max_info[1], local_domains[p].max_face_nodeid);
        local_domains[p].max_cell_nodeid = std::max(max_info[2], local_domains[p].max_cell_nodeid);

        auto &map_cells = local_domains[p].map_cells;
        auto &map_nodes = local_domains[p].map_nodes;
        auto &map_halos = local_domains[p].map_halos;
        auto &map_halo_int = local_domains[p].map_halo_int;
        auto &set_phyid = local_domains[p].set_phyids;
        auto &set_halo_phyid_neighsub = local_domains[p].set_halo_phyid_neighsub;

        // local cells

        map_cells[i] = (int32_t)map_cells.size();

        // Determine max_cell_halonid, Create HaloCellMap, Create halo_interior_map, nb_node_halos
        int32_t nb_cell_halonid = 0;
        const auto the_cell = cells->sub_array(i);
        for (int32_t j = 0; j < the_cell.last(); j++) {
            const int32_t nodeid = the_cell.get(j);
            int32_t nb_node_halonid = 0;

            const auto the_node_cellid = node_cellid->sub_array(nodeid);
            for (int32_t k = 0; k < the_node_cellid.last(); k++) {
              const int32_t n_cellid = the_node_cellid.get(k);
              const int32_t part_n_cellid = part_vert->get(n_cellid);

              if (p != part_n_cellid) {
                nb_node_halonid += 1;

                //  halos
                if (map_halos.find(n_cellid) == map_halos.end()) {
                  map_halos[n_cellid] = (int32_t)map_halos.size();
                }

                //  halo_interior
                if (map_halo_int.find(part_n_cellid) == map_halo_int.end()) {
                  map_halo_int[part_n_cellid] = std::vector<int32_t>();
                }

                std::vector<int32_t> &vec = map_halo_int[part_n_cellid];
                if (vec.empty() or vec.back() != i) {
                  vec.push_back(i); // append haloint_cell `i` to halo interiors connected to neighbor part `part_n_cellid`
                }

                // nb_cell_halonid
                if (n_cellid != i and i_visited[n_cellid] != i) {
                  // allow visiting n_cellid only once for the current cell `i`
                  i_visited[n_cellid] = i;
                  nb_cell_halonid += 1;
                }
              }
            }

            if (map_nodes.find(nodeid) == map_nodes.end()) {
                local_domains[p].nb_node_halos += nb_node_halonid;
                if (nb_node_halonid != 0) {
                    local_domains[p].nb_node_halos += 2;
                }
                map_nodes[nodeid] = (int32_t)map_nodes.size();
            }

            // max_node_halophyid, set_phyid, set_halo_phyid_neighsub
            int32_t nb_node_halophyid = 0;
            auto the_node_phyid = node_phyid->sub_array(nodeid);
            for (int32_t k = 0; k < the_node_phyid.last(); k++) {
                const int32_t phy_id = the_node_phyid.get(k);
                const int32_t phy_id_part = part_phyid[phy_id];
                if (set_phyid.count(phy_id) == 0) {
                    set_phyid.insert(phy_id);
                }
                if (p != phy_id_part) {
                    nb_node_halophyid += 1;
                    set_halo_phyid_neighsub.insert(phy_id_part);
                }
            }
            local_domains[p].max_node_halophyid = std::max(nb_node_halophyid, local_domains[p].max_node_halophyid);
        }
        local_domains[p].max_cell_halonid = std::max(nb_cell_halonid, local_domains[p].max_cell_halonid);
    }
}




// #################################################################
// 2. _create_partition_tables
// #################################################################

static void _create_locals(const int32_t p,
PyArray<int32_t, 2> *cells,
PyArray<double, 2> *nodes,
PyArray<int8_t, 1> *cells_type,
PyArray<int32_t, 2> *node_cellid,
PyArray<int32_t, 2> *node_phyid,
std::vector<int32_t> &part_phyid,
PyArray<int32_t, 2> *phy_faces,
PyArray<int32_t, 1> *phy_faces_name,
PyArray<int32_t, 1> *part_vert,
std::vector<int32_t> &vec_node_oldname,
LocalDomainStruct &local_domain) {

    auto &map_cells = local_domain.map_cells;
    auto &map_nodes = local_domain.map_nodes;
    auto &map_halos = local_domain.map_halos;
    auto &map_phy_faces = local_domain.map_phy_faces;
    auto &map_halo_int = local_domain.map_halo_int;
    auto l_cells = local_domain.cells;
    auto l_cell_loctoglob = local_domain.cell_loctoglob;
    auto l_cells_type = local_domain.cells_type;
    auto l_nodes = local_domain.nodes;
    auto l_node_loctoglob = local_domain.node_loctoglob;
    auto l_node_oldname = local_domain.node_oldname;
    auto l_node_halos = local_domain.node_halos;
    auto l_node_halophyid = local_domain.node_halophyid;
    auto l_phy_faces = local_domain.phy_faces;
    auto l_phy_faces_name = local_domain.phy_faces_name;
    auto l_halo_neighsub = local_domain.halo_neighsub;
    auto l_halo_halosint = local_domain.halo_halosint;
    auto l_halo_halosext = local_domain.halo_halosext;

    auto &map_phyids = local_domain.map_phyids;
    auto &vec_phyids = local_domain.vec_phyids;
    auto phyid_recv = local_domain.phyid_recv;
    auto phyid_recv_part_size = local_domain.phyid_recv_part_size;

    // #########################################################
    // l_cells, l_cells_type, l_cell_loctoglob
    // #########################################################

    for (const auto &item : map_cells) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        l_cells_type->get(l_id) = cells_type->get(g_id); // cell_type
        l_cell_loctoglob->get(l_id) = g_id; // cell_loctoglob

        const auto the_cells = cells->sub_array(g_id);
        auto the_l_cells = l_cells->sub_array(l_id);
        for (int32_t j = 0; j < the_cells.last(); j++) {
            const int32_t nodeid = the_cells.get(j);
            the_l_cells.get(j) = map_nodes[nodeid];
        }
        the_l_cells.last() = the_cells.last();
    }

    // #########################################################
    // l_nodes, l_node_loctoglob, l_node_oldname, l_node_halos, l_node_halophyid, max_node_haloid
    // #########################################################
    int32_t halos_counter = 0;
    for (const auto &item: map_nodes) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        l_node_loctoglob->get(l_id) = g_id;
        l_node_oldname->get(l_id) = vec_node_oldname[g_id];

        const auto the_nodes = nodes->sub_array(g_id);
        const auto the_l_nodes = l_nodes->sub_array(l_id);
        for (int32_t j = 0; j < nodes->shape[1]; j++) {
            the_l_nodes.get(j) = the_nodes.get(j);
        }

        int32_t node_counter = -1;
        const auto the_node_cellid = node_cellid->sub_array(g_id);
        for (int32_t j = 0; j < the_node_cellid.last(); j++) {
            const int32_t neighbor_cell = the_node_cellid.get(j);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);
            if (p != neighbor_part) {
                if (node_counter == -1) {
                    l_node_halos->get(halos_counter) = l_id;
                    l_node_halos->get(halos_counter + 1) = 0;
                    node_counter = halos_counter + 1;
                    halos_counter += 2;
                }
                l_node_halos->get(halos_counter) = map_halos[neighbor_cell];
                halos_counter += 1;
                l_node_halos->get(node_counter) += 1;
            }
        }

        if (node_counter != -1) {
            local_domain.max_node_haloid = std::max(l_node_halos->get(node_counter), local_domain.max_node_haloid);
        }

        // l_node_halophyid
        const auto the_node_phyid = node_phyid->sub_array(g_id);
        auto the_l_node_halophyid = l_node_halophyid->sub_array(l_id);
        int32_t counter = 0;
        for (int32_t j = 0; j < the_node_phyid.last(); j++) {
            const int32_t neighbor_phyid = the_node_phyid.get(j);
            const int32_t neighbor_part = part_phyid[neighbor_phyid];
            if (p != neighbor_part) {
                the_l_node_halophyid.get(counter) = map_phyids[neighbor_phyid];
                counter++;
            }
        }
        the_l_node_halophyid.last() = counter;
    }

    // #########################################################
    // l_phy_faces, l_phy_faces_name
    // #########################################################

    for (const auto &item: map_phy_faces) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        l_phy_faces_name->get(l_id) = phy_faces_name->get(g_id);

        const auto the_phy_faces = phy_faces->sub_array(g_id);
        auto the_l_phy_faces = l_phy_faces->sub_array(l_id);
        for (int32_t j = 0; j < the_phy_faces.last(); j++) {
            const int32_t nodeid = the_phy_faces.get(j);
            the_l_phy_faces.get(j) = map_nodes[nodeid];
        }
        the_l_phy_faces.last() = the_phy_faces.last();
    }

    // #########################################################
    // l_halo_neighsub, l_halo_halosint, l_halo_halosext
    // #########################################################

    int32_t neighsub_counter = 0;
    int32_t halosint_counter = 0;
    for (const auto &item : map_halo_int) {
        const int32_t partition = item.first;

        const auto &vect = map_halo_int[partition];
        l_halo_neighsub->sub_array(0).get(neighsub_counter) = partition;
        l_halo_neighsub->sub_array(1).get(neighsub_counter) = (int32_t)vect.size();
        neighsub_counter += 1;
        for (const auto interior_cell: vect) {
            l_halo_halosint->get(halosint_counter) = interior_cell;
            halosint_counter += 1;
        }
    }

    for (const auto &item: map_halos) {
        const int32_t g_id = item.first;
        const int32_t l_id = item.second;

        auto the_l_halo_halosext = l_halo_halosext->sub_array(l_id);
        the_l_halo_halosext.get(0) = g_id;
        auto the_cells = cells->sub_array(g_id);
        for (int32_t j = 0; j < the_cells.last(); j++) {
            const int32_t nodeid = the_cells.get(j);
            the_l_halo_halosext.get(j + 1) = nodeid;
        }
        the_l_halo_halosext.last() = the_cells.last() + 1;
    }

    // #########################################################
    // phyid_recv => [phyid of p_a, ..., phyid of p_b, ...]
    // phyid_recv_part_size => [partition Id, size, ...]
    // #########################################################

    int32_t old_part = -1;
    int32_t counter = 0;
    bool p_has_halo_phyid = false;
    for (int32_t i = 0; i < vec_phyids.size(); i++) {
        const int32_t g_id = vec_phyids[i];
        const int32_t part = part_phyid[g_id];
        phyid_recv->get(i) = g_id;
        if (p == part) {
            p_has_halo_phyid = true;
            phyid_recv->get(i) = map_phy_faces[g_id]; // transform phyid to local for p == part
        }
        if (old_part != part) {
            phyid_recv_part_size->get(counter) = part;
            phyid_recv_part_size->get(counter + 1) = 0;
            old_part = part;
            counter += 2;
        }
        phyid_recv_part_size->get(counter - 1) += 1;
    }
    if (!p_has_halo_phyid) {
        phyid_recv_part_size->get(counter) = p;
        phyid_recv_part_size->get(counter + 1) = 0;
    }
}


static void _create_phyid_send(LocalDomainStruct *local_domains, const int32_t n) {
    // #########################################################
    // phyid_send => [partition_id, size, indices point to phyid_recv_part_size, ...]
    // #########################################################

    for (int32_t p = 0; p < n; p++) {
        auto &vec_phyids = local_domains[p].vec_phyids;
        auto phyid_recv_part_size = local_domains[p].phyid_recv_part_size;
        int32_t counter = 0;

        for (int32_t i = 0; i < phyid_recv_part_size->shape[0]; i += 2) {
            const int32_t part = phyid_recv_part_size->get(i);
            const int32_t size = phyid_recv_part_size->get(i + 1);
            if (part != p) {
                auto &list_phyid_send = local_domains[part].list_phyid_send;
                auto &map_phyids = local_domains[part].map_phyids;
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

    for (int32_t p = 0; p < n; p++) {
        auto &list_phyid_send = local_domains[p].list_phyid_send;
        local_domains[p].phyid_send = new PyArray<int32_t, 1>(make_npy_dims(list_phyid_send.size()));
        for (int32_t i = 0; i < list_phyid_send.size(); i++) {
            local_domains[p].phyid_send->get(i) = list_phyid_send[i];
        }
    }
}

static void _create_partition_tables(
LocalDomainStruct *local_domains,
PyArray<int32_t, 2> *cells,
PyArray<double, 2> *nodes,
PyArray<int8_t, 1> *cells_type,
PyArray<int32_t, 2> *node_cellid,
PyArray<int32_t, 2> *node_phyid,
std::vector<int32_t> &part_phyid,
PyArray<int32_t, 2> *phy_faces,
PyArray<int32_t, 1> *phy_faces_name,
PyArray<int32_t, 1> *part_vert,
std::vector<int32_t> &vec_node_oldname,
const int32_t n) {
    for (int32_t p = 0; p < n; p++) {
        local_domains[p]._create_tables(nodes, part_phyid);
        _create_locals(p, cells, nodes, cells_type, node_cellid, node_phyid, part_phyid, phy_faces, phy_faces_name, part_vert, vec_node_oldname, local_domains[p]);
    }
    _create_phyid_send(local_domains, n);
}

// #################################################################
// 3. Return
// #################################################################

static PyObject *get_result_as_py_list(LocalDomainStruct *local_domains, const int32_t nb_parts) {
    PyObject *py_list_result = PyList_New(nb_parts);
    if (!py_list_result) {
        throw std::bad_alloc();
    }
    for (int i = 0; i < nb_parts; i++) {
        local_domains[i].create_tuple(); // create local_domains[i].tuple_res
        PyList_SET_ITEM(py_list_result, i, local_domains[i].tuple_res);

        // The ownership transferred to the list.
        local_domains[i].tuple_res = nullptr;
    }

    return py_list_result;
}


PyObject * create_local_domains(
LocalDomainStruct *local_domains,
PyArray<int32_t, 1> *part_vert,
PyArray<int32_t, 2> *node_cellid,
PyArray<int32_t, 2> *node_phyid,
PyArray<int32_t, 2> *cells,
PyArray<int8_t, 1> *cells_type,
PyArray<double, 2> *nodes,
PyArray<int32_t, 2> *phy_faces,
PyArray<int32_t, 1> *phy_faces_name,
const int32_t nb_parts) {
    std::vector<int32_t> i_visited(cells->shape[0], -1);
    std::vector<int32_t> vec_node_oldname(nodes->shape[0]);
    std::vector<int32_t> intersect_cell(2);
    std::vector<int32_t> boundary_cells(phy_faces->shape[0]); //cells that has at least one physical face attached to it
    std::vector<int32_t> part_phyid(phy_faces->shape[0]);

    print_instant("1. _create_sub_domains\n");
    _create_sub_domains(part_vert, node_cellid, node_phyid, cells, cells_type, phy_faces, phy_faces_name, local_domains, i_visited, vec_node_oldname, intersect_cell, boundary_cells, part_phyid);

    print_instant("2. _create_partition_tables\n");
    _create_partition_tables(local_domains, cells, nodes, cells_type, node_cellid, node_phyid, part_phyid, phy_faces, phy_faces_name, part_vert, vec_node_oldname, nb_parts);

    print_instant("3. Return\n");
    PyObject *py_list_result = get_result_as_py_list(local_domains, nb_parts);

    return py_list_result;
}