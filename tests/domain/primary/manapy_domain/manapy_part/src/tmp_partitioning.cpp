#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"
#include "LocalDomainStruct.h"

/*
 * @local_node_cellid TODO probabily i will not need those i will need only nodes_is_boundary
 * @local_b_node
 * @nb_node_halos => node_cellid - local_node_cellid
 */

/*
 * @nodes
 * @cells
 * @cells_type
 * @cell_loctoglob
 * @node_loctoglob
 * @node_halos
 * @halo_neighsub
 * @halo_halosext
 * @halo_halosint
 * @max_node_haloid,
 * @max_cell_nodeid,
 * @max_cell_faceid,
 * @max_face_nodeid,
 * @max_cell_halonid
 */




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
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *cells,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<int32_t, 1> *node_loctoglob,
    PyArray<int32_t, 1> *cell_loctoglob,
    PyArray<int32_t, 2> *local_node_cellid,
    std::vector<int32_t> &local_b_nodes,
    const int32_t max_cell_nodeid,
    const int32_t nb_node_halos,
    const int32_t p) {

    // #########################################################
    // max_node_haloid, l_node_halos
    // #########################################################
    auto l_node_halos = new PyArray<int32_t, 1>(make_npy_dims(nb_node_halos));

    std::map<int32_t, int32_t> map_halos;
    std::map<int32_t, std::set<int32_t> > map_int_halos;
    int32_t halos_counter = 0;
    int32_t max_node_haloid = -1;

    for (const int32_t local_node_id : local_b_nodes) {
        const int32_t g_index = node_loctoglob->get(local_node_id);
        int32_t node_counter = -1;
        auto sub_node_cellid = node_cellid->sub_array(g_index);
        for (int32_t i = 0; i < sub_node_cellid.last(); i++) {
            const int32_t neighbor_cell = sub_node_cellid.get(i);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);
            if (neighbor_part != p) {
                //TODO unique halo_halosext
                if (map_halos.find(neighbor_cell) == map_halos.end()) {
                    map_halos[neighbor_cell] = (int32_t)map_halos.size();
                }

                if (map_int_halos.find(neighbor_part) == map_int_halos.end()) {
                    map_int_halos[neighbor_part] = std::set<int32_t>();
                }
                auto &set_int_halos = map_int_halos[neighbor_part];
                auto sub_local_node_cellid = local_node_cellid->sub_array(local_node_id);
                //TODO skep the same n_cellid_part for a local_node
                for (int32_t j = 0; j < sub_local_node_cellid.last(); j++) {
                    const int32_t local_n_cellid = sub_local_node_cellid.get(j);
                    const int32_t cell_g_index = cell_loctoglob->get(local_n_cellid);
                    set_int_halos.insert(cell_g_index);
                }



                if (node_counter == -1) {
                    l_node_halos->get(halos_counter) = local_node_id;
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
            max_node_haloid = std::max(l_node_halos->get(node_counter), max_node_haloid);
        }
    }


    // #########################################################
    // l_halo_halosext
    // #########################################################

    PyArray<int32_t, 2> *l_halo_halosext = new PyArray<int32_t, 2>(make_npy_dims(map_halos.size(), max_cell_nodeid + 2));

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
    // l_halo_neighsub, l_halo_halosint
    // #########################################################

    int32_t nb_halos_int = 0;
    for (const auto &item : map_int_halos) {
        nb_halos_int += (int32_t)map_int_halos[item.first].size();
    }

    PyArray<int32_t, 2> *l_halo_neighsub = new PyArray<int32_t, 2>(make_npy_dims(2, map_int_halos.size()));
    PyArray<int32_t, 1> *l_halo_halosint = new PyArray<int32_t, 1>(make_npy_dims(nb_halos_int));

    int32_t neighsub_counter = 0;
    int32_t halosint_counter = 0;
    for (const auto &item : map_int_halos) {
        const int32_t partition = item.first;

        const auto &set = map_int_halos[partition];
        l_halo_neighsub->sub_array(0).get(neighsub_counter) = partition;
        l_halo_neighsub->sub_array(1).get(neighsub_counter) = (int32_t)set.size();
        neighsub_counter += 1;
        for (const auto interior_cell: set) {
            l_halo_halosint->get(halosint_counter) = interior_cell;
            halosint_counter += 1;
        }
    }
}

void devide(
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<double, 2> *nodes,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    const int32_t nb_parts
    ) {

    print_instant("Start devide\n");
    time_it(""); // start time

    std::vector<int32_t> parts; // temporarily vector to store node neighboring parts
    std::vector<int32_t> parts_counter(nb_parts, 0); // for a fixed node `i` determine the number of neighbors for each part
    std::vector<int32_t> local_nodes_counter(nb_parts, 0); // determine the number of nodes for each parts
    std::vector<int32_t> local_nodes_max_neighbors(nb_parts, 0); // determine the max neighboring cellid for each part
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

            //update the max node neighbors
            local_nodes_max_neighbors[part] = std::max(local_nodes_max_neighbors[part], parts_counter[part]);

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
    // creating: local_nodes, local_node_loctoglob, local_node_cellid, local_b_nodes, nb_node_halos
    // #########################################################

    std::vector<PyArray<double, 2> *> local_nodes(nb_parts, nullptr);
    std::vector<PyArray<int32_t, 1> *> local_node_loctoglob(nb_parts, nullptr);
    std::vector<PyArray<int32_t, 2> *> local_node_cellid(nb_parts, nullptr);
    std::vector<std::vector<int32_t> > local_b_nodes(nb_parts);
    std::vector<int32_t> local_nb_node_halos(nb_parts, 0);
    std::vector<std::map<int32_t, int32_t> > vec_map_nodes(nodes->shape[0]); // for every g_node store local id of a certain partition
    std::vector<bool> node_is_boundary(nodes->shape[0], false);

    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_nodes = local_nodes_counter[i];
        const int32_t nb_b_nodes = local_boundary_nodes_counter[i];
        const int32_t max_node_cellid = local_nodes_max_neighbors[i];

        local_nodes[i] = new PyArray<double, 2>(make_npy_dims(nb_nodes, nodes->shape[1]));
        local_node_loctoglob[i] = new PyArray<int32_t, 1>(make_npy_dims(nb_nodes));
        local_node_cellid[i] = new PyArray<int32_t, 2>(make_npy_dims(nb_nodes, max_node_cellid + 1), true);
        local_b_nodes[i].resize(nb_b_nodes);
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
            auto sub_local_node_cellid = local_node_cellid[neighbor_part]->sub_array(local_nodeid);

            if (parts_counter[neighbor_part] == 0) {
                parts.push_back(neighbor_part);

                //* assign node_loctoglob
                local_node_loctoglob[neighbor_part]->get(local_nodeid) = i;
                vec_map_nodes[i][neighbor_part] = local_nodeid;
            }

            //* assign local_node_cellid
            int32_t &size = sub_local_node_cellid.last();
            sub_local_node_cellid.get(size) = neighbor_cell;
            size++; // increment the reference



            //increment parts_counter for a neighbor_part
            parts_counter[neighbor_part]++;


            //* assign local_nodes
            const auto sub_l_nodes = local_nodes[neighbor_part]->sub_array(local_nodeid);
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
                local_b_nodes[part][counter] = local_nodeid;
                counter++;

                //* assign local_nodes_is_boundary
                node_is_boundary[i] = true;
            }

            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;

            //* assign nb_node_halos
            local_nb_node_halos[part] = sub_node_cellid.last() - parts_counter[part];

            // reset parts_counter
            parts[j] = 0;
            parts_counter[part] = 0;
        }

        // reset the size of the vector
        parts.clear();
    }


    // #########################################################
    // creating:  local_cells, local_cells_type, local_cell_loctoglob, local_max_cell_nodeid, local_max_cell_faceid, local_max_face_nodeid, local_max_cell_halonid
    // #########################################################

    std::vector<int32_t> local_nb_cells(nb_parts, 0);
    std::vector<int32_t> local_max_cell_faceid(nb_parts, 0);
    std::vector<int32_t> local_max_face_nodeid(nb_parts, 0);
    std::vector<int32_t> local_max_cell_nodeid(nb_parts, 0);
    std::vector<int32_t> local_max_cell_halonid(nb_parts, 0);
    std::vector<int32_t> i_visited(cells->shape[0], -1);

    for (int32_t i = 0; i < cells->shape[0]; i++) {
        const int32_t part = part_vert->get(i);
        const int8_t cell_type = cells_type->get(i);
        const auto max_info = _get_max_info(cell_type);

        local_max_cell_faceid[part] = std::max(max_info[0], local_max_cell_faceid[part]);
        local_max_face_nodeid[part] = std::max(max_info[1], local_max_face_nodeid[part]);
        local_max_cell_nodeid[part] = std::max(max_info[2], local_max_cell_nodeid[part]);
        local_nb_cells[part]++;
    }

    std::vector<PyArray<int32_t, 2> *> local_cells(nb_parts, nullptr);
    std::vector<PyArray<int8_t, 1> *> local_cells_type(nb_parts, nullptr);
    std::vector<PyArray<int32_t, 1> *> local_cell_loctoglob(nb_parts, nullptr);
    std::vector<std::map<int32_t, std::set<int32_t> > > local_map_int_halos(nb_parts);

    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_cells = local_nb_cells[i];
        const int32_t max_cell_nodeid = local_max_cell_nodeid[i];

        local_cells[i] = new PyArray<int32_t, 2>(make_npy_dims(nb_cells, max_cell_nodeid + 1));
        local_cells_type[i] = new PyArray<int8_t, 1>(make_npy_dims(nb_cells));
        local_cell_loctoglob[i] = new PyArray<int32_t, 1>(make_npy_dims(nb_cells));
    }

    std::fill(local_nb_cells.begin(), local_nb_cells.end(), 0);
    for (int32_t g_id = 0; g_id < cells->shape[0]; g_id++) {
        const int32_t part = part_vert->get(g_id);
        const int32_t l_id = local_nb_cells[part];
        const int8_t cell_type = cells_type->get(g_id);
        auto &map_int_halos = local_map_int_halos[part];

        //* assign local_cells_type
        local_cells_type[part]->get(l_id) = cell_type;

        //* assign l_cell_loctoglob
        local_cell_loctoglob[part]->get(l_id) = g_id;

        //* assign local_cells
        const auto sub_cells = cells->sub_array(g_id);
        auto sub_l_cells = local_cells[part]->sub_array(l_id);
        int32_t nb_cell_halonid = 0;
        for (int32_t j = 0; j < sub_cells.last(); j++) {
            const int32_t nodeid = sub_cells.get(j);
            const int32_t local_nodeid = vec_map_nodes[nodeid][part];
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

                        // if (map_int_halos.find(neighbor_part) == map_int_halos.end()) {
                        //     map_int_halos[neighbor_part] = std::set<int32_t>();
                        // }
                        // auto &set_int_halos = map_int_halos[neighbor_part];
                        // set_int_halos.insert(g_id);
                    }
                }
            }
        }
        sub_l_cells.last() = sub_cells.last();

        //* assign local_max_cell_halonid
        local_max_cell_halonid[part] = std::max(local_max_cell_halonid[part], nb_cell_halonid);


        local_nb_cells[part]++;
    }

    // #########################################################
    // creating:  local_cells, local_cells_type, local_cell_loctoglob, local_max_cell_nodeid, local_max_cell_faceid, local_max_face_nodeid, local_max_cell_halonid
    // #########################################################

    time_it("devide");
    // for (int32_t p = 0; p < nb_parts; p++) {
    //     auto l_node_loctoglob = local_node_loctoglob[p];
    //     auto l_cell_loctoglob = local_cell_loctoglob[p];
    //     auto l_node_cellid = local_node_cellid[p];
    //     auto &l_b_nodes = local_b_nodes[p];
    //     const int32_t max_cell_nodeid = local_max_cell_nodeid[p];
    //     const int32_t nb_node_halos = local_nb_node_halos[p];
    // time_it("");
    //     create_halos(part_vert, cells, node_cellid, l_node_loctoglob, l_cell_loctoglob, l_node_cellid, l_b_nodes, max_cell_nodeid, nb_node_halos, p);
    // time_it("create_halos");
    // }
}




