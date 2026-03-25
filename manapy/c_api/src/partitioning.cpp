#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"

/*
 * public functions:
 * create_sub_domains
 *
 * private:
 * create_halos
 * create_phy
 * create_phyid_send
 * loop_through_nodes
 * loop_through_physical_faces
 * loop_through_cells
 * get_result_as_py_list
 */

static void create_halos(
    LocalDomainStruct *ld,
    PyArray<int32_t, 2> *cells,
    PyArray<fdx_t, 2> *nodes,
    std::vector<int32_t> &vec_cell_to_halo,
    const int32_t p,
    const int32_t dim
    ) {

    /*
     * RETURN: (description at LocalDomainStruct.h)
     * ld[x].max_node_haloid
     * ld[x].node_halos
     * ld[x].halo_halosext
     * ld[x].halo_neighsub
     * ld[x].halo_halosint
     * ld[x].halo_centvol
     */

    //variables needed as read only
    const auto &l_map_int_halos = ld[p].map_int_halos;
    const auto &vec_node_halos = ld[p].vec_node_halos;
    const int32_t nb_nodes = static_cast<int32_t>(ld[p].nodes->shape[0]);

    //write
    int32_t &l_max_node_haloid = ld[p].max_node_haloid;
    l_max_node_haloid = -1;

    // #########################################################
    // halo_halosext, local_max_node_haloid, local_node_halos
    // #########################################################
    // The concept is that Halos of partition P is the Concatenation of Interiors of the neighbor parts that are connected to P
    // Global Id of the Interiors of the other partitions can't be duplicated (a cell can't be shared with two paritions)

    //local
    std::vector<int32_t> vec_max(nb_nodes, 0);
    int32_t nb_halos = 0;

    // Count the number of received halos
    for (const auto &item: ld[p].map_int_halos) {
        const auto neighbor = item.first;
        const auto &halos_int = ld[neighbor].map_int_halos[p];
        nb_halos += static_cast<int32_t>(halos_int.size());
    }

    // Allocate of halosext
    ld[p].halo_halosext = new PyArray<int32_t, 2>(make_npy_dims(nb_halos, ld[p].max_halo_cell_nodeid + 2));
    const auto *l_halo_halosext = ld[p].halo_halosext;


    int32_t counter = 0;
    for (const auto &item: ld[p].map_int_halos) {
        const int32_t neighbor = item.first;
        const auto &halos_int = ld[neighbor].map_int_halos[p]; // interiors of the neighbor partition.

        for (const auto &halo: halos_int) {
            const int32_t g_id = ld[neighbor].cell_loctoglob->get(halo);

            //*** vec_cell_to_halo
            vec_cell_to_halo[g_id] = counter;

            //*** start halo_halosext
            auto sub_l_halo_halosext = l_halo_halosext->sub_array(counter);
            sub_l_halo_halosext.get(0) = g_id;
            auto sub_cells = cells->sub_array(g_id);
            for (int32_t j = 0; j < sub_cells.last(); j++) {
                const int32_t nodeid = sub_cells.get(j);
                sub_l_halo_halosext.get(j + 1) = nodeid;
            }
            sub_l_halo_halosext.last() = sub_cells.last() + 1;
            //*** end halo_halosext

            counter++;
        }
    }
    // #########################################################

    ld[p].node_halos = new PyArray<int32_t, 1>(make_npy_dims(vec_node_halos.size()));
    const auto l_node_halos = ld[p].node_halos;

    //*** node_halos, max_node_haloid
    for (int32_t i = 0; i < vec_node_halos.size(); i=i+2) {
        // copy node_id
        l_node_halos->get(i) = vec_node_halos[i];
        // determine max_node_haloid
        l_max_node_haloid = std::max(l_max_node_haloid, ++vec_max[vec_node_halos[i]]);
    }
    for (int32_t i = 1; i < vec_node_halos.size(); i=i+2) {
        // copy the index that will point to halosext
        l_node_halos->get(i) = vec_cell_to_halo[vec_node_halos[i]];
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

        //***halo_neighsub
        l_halo_neighsub->sub_array(0).get(neighsub_counter) = partition;
        l_halo_neighsub->sub_array(1).get(neighsub_counter) = (int32_t)set.size();
        neighsub_counter += 1;
        for (const auto interior_cell: set) {
            //*** halo_halosint
            l_halo_halosint->get(halosint_counter) = interior_cell;
            halosint_counter += 1;
        }
    }

    // #########################################################
    // l_halo_centvol
    // #########################################################
    //*** start halo_centvol
    ld[p].halo_centvol = new PyArray<fdx_t, 2>(make_npy_dims(nb_halos, 4)); // x, y, z, area
    if (dim == 2)
        compute_halo_cell_center_area_2d(ld[p].halo_halosext, nodes, ld[p].halo_centvol);
    else if (dim == 3)
        compute_halo_cell_center_volume_3d(ld[p].halo_halosext, nodes, ld[p].halo_centvol);
    //*** end halo_centvol
}

static void create_phy(
    LocalDomainStruct *ld,
    const int32_t p,
    const PyArray<int32_t, 1> *phy_faces_name,
    const PyArray<int32_t, 2> *phy_faces,
    const std::vector<int32_t> &vec_node_oldname,
    const std::vector<std::map<int32_t, int32_t> > &vec_map_nodes
    ) {

    /*
     * RETURN: (description at LocalDomainStruct.h)
     * ld[x].phyid_neighbor
     * ld[x].phyid_recv
     * ld[x].phyid_send
     * ld[x].node_halophyid
     * ld[x].cell_halophyid
     * ld[x].node_oldname
     * ld[x].phy_faces
     * ld[x].phy_faces_name
     * ld[x].max_cell_halophyid
     * ld[x].max_node_halophyid
     */

    //read only variables
    // Description in the header file
    const auto &l_map_phy_faces = ld[p].map_phyid;
    const auto max_phy_face_nodeid = ld[p].max_phy_face_nodeid;
    const auto l_node_loctoglob = ld[p].node_loctoglob;
    const auto l_nb_nodes = static_cast<int32_t>(l_node_loctoglob->shape[0]);
    const auto &l_map_phyid_recv = ld[p].map_phyid_recv;
    const auto &l_map_node_halophyid = ld[p].map_node_halophyid;
    const auto &l_map_cell_halophyid = ld[p].map_cell_halophyid;

    /// Write variables
    std::map<int32_t, int32_t> map_halophyid; ///< Map halophyid to its location inside phyid_recv

    // #########################################################
    // phyid_neighbor phyid_recv phyid_send node_halophyid cell_halophyid
    // #########################################################

    // Assumption: The neighbors to send are equal to those to receive.
    int32_t neighbor_size = 0;
    int32_t recv_size = 0;
    int32_t send_size = 0;
    int32_t node_halophyid_size = 0;
    int32_t cell_halophyid_size = 0;

    //*** Compute sizes for phyid_neighbor phyid_recv phyid_send node_halophyid
    for (const auto &[neighbor_part_id, set_halophyid]: l_map_phyid_recv) {
        recv_size += static_cast<int32_t>(set_halophyid.size());
        send_size += static_cast<int32_t>(ld[neighbor_part_id].map_phyid_recv[p].size());
        neighbor_size++;
    }
    for (const auto& [_, set_node_halophyid] : l_map_node_halophyid) {
        const auto size = static_cast <int32_t>(set_node_halophyid.size());
        ld[p].max_node_halophyid = std::max(ld[p].max_node_halophyid, size);
        node_halophyid_size += size + 2; // 1 for node, 1 for size
    }
    for (const auto& [_, set_cell_halophyid] : l_map_cell_halophyid) {
        const auto size = static_cast <int32_t>(set_cell_halophyid.size());
        ld[p].max_cell_halophyid = std::max(ld[p].max_cell_halophyid, size);
        cell_halophyid_size += size + 2; // 1 for cell, 1 for size
    }
    //*** End Compute sizes

    //*** Allocate
    // Description in the header file
    auto *py_phyid_neighbor = new PyArray<int32_t, 2>(make_npy_dims(neighbor_size, 3));
    auto *py_phyid_recv = new PyArray<int32_t, 1>(make_npy_dims(recv_size));
    auto *py_phyid_send = new PyArray<int32_t, 1>(make_npy_dims(send_size));
    auto *py_node_halophyid = new PyArray<int32_t, 1>(make_npy_dims(node_halophyid_size));
    auto *py_cell_halophyid = new PyArray<int32_t, 1>(make_npy_dims(cell_halophyid_size));
    ld[p].phyid_neighbor = py_phyid_neighbor;
    ld[p].phyid_recv = py_phyid_recv;
    ld[p].phyid_send = py_phyid_send;
    ld[p].node_halophyid = py_node_halophyid;
    ld[p].cell_halophyid = py_cell_halophyid;
    //*** End Allocate

    //*** start phyid_neighbor, phyid_recv, phyid_send
    neighbor_size = 0;
    recv_size = 0;
    send_size = 0;

    // the problem is: if a part does not receive from a neighbor anything, it will cause the neighbor to not be appended therefor not send anything.
    for (const auto& [neighbor_part_id, set_halophyid]: l_map_phyid_recv) {
        //set_halophyid: Elements that will be received
        //set_intphyid: Elements that will be sent
        const auto &set_intphyid = ld[neighbor_part_id].map_phyid_recv.at(p);

        auto a = ld[p].map_phyid_recv.at(neighbor_part_id);

        // Create phyid_neighbor
        py_phyid_neighbor->get2(neighbor_size, 0) = neighbor_part_id;
        py_phyid_neighbor->get2(neighbor_size, 1) = static_cast<int32_t>(set_intphyid.size());
        py_phyid_neighbor->get2(neighbor_size, 2) = static_cast<int32_t>(set_halophyid.size());
        neighbor_size++;

        // Create phyid_recv
        for (const int32_t halophyid: set_halophyid) {
            py_phyid_recv->get(recv_size) = halophyid;
            //py_phyid_recv->get(recv_size) = ld[neighbor_part_id].map_phyid.at(halophyid);

            map_halophyid[halophyid] = recv_size;
            recv_size++;
        }

        // Create phyid_send
        for (const int32_t intphyid: set_intphyid) {
            py_phyid_send->get(send_size) = l_map_phy_faces.at(intphyid);
            send_size++;
        }
    }
    //*** end phyid_neighbor, phyid_recv, phyid_send

    //*** start node_halophyid
    node_halophyid_size = 0;
    for (const auto& [local_node_id, set_node_halophyid] : l_map_node_halophyid) {
        py_node_halophyid->get(node_halophyid_size) = local_node_id;
        node_halophyid_size++;
        py_node_halophyid->get(node_halophyid_size) = static_cast<int32_t>(set_node_halophyid.size());
        node_halophyid_size++;
        for (const int32_t halophyid: set_node_halophyid) {
            py_node_halophyid->get(node_halophyid_size) = map_halophyid.at(halophyid);
            node_halophyid_size++;
        }
    }
    //*** end node_halophyid

    //*** start cell_halophyid
    cell_halophyid_size = 0;
    for (const auto& [local_cell_id, set_cell_halophyid] : l_map_cell_halophyid) {
        py_cell_halophyid->get(cell_halophyid_size) = local_cell_id;
        cell_halophyid_size++;
        py_cell_halophyid->get(cell_halophyid_size) = static_cast<int32_t>(set_cell_halophyid.size());
        cell_halophyid_size++;
        for (const int32_t halophyid: set_cell_halophyid) {
            py_cell_halophyid->get(cell_halophyid_size) = map_halophyid.at(halophyid);
            cell_halophyid_size++;
        }
    }
    //*** end cell_halophyid

    // #########################################################
    // l_node_oldname, l_node_halophyid
    // #########################################################

    ld[p].node_oldname = new PyArray<int32_t, 1>(make_npy_dims(l_nb_nodes));
    auto l_node_oldname = ld[p].node_oldname;

    for (int32_t l_id = 0; l_id < l_nb_nodes; l_id++) {
        const int32_t g_id = l_node_loctoglob->get(l_id);

        //*** node_oldname
        l_node_oldname->get(l_id) = vec_node_oldname[g_id];
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

        //*** phy_faces_name
        l_phy_faces_name->get(l_id) = phy_faces_name->get(g_id);

        //*** start phy_faces
        const auto sub_phy_faces = phy_faces->sub_array(g_id);
        auto sub_l_phy_faces = l_phy_faces->sub_array(l_id);
        for (int32_t j = 0; j < sub_phy_faces.last(); j++) {
            const int32_t nodeid = sub_phy_faces.get(j);
            sub_l_phy_faces.get(j) = vec_map_nodes[nodeid].at(p);
        }
        sub_l_phy_faces.last() = sub_phy_faces.last();
        //*** end phy_faces
    }
}


static void loop_through_nodes(
    LocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<fdx_t, 2> *nodes,
    std::vector<bool> &node_is_boundary,
    std::vector<std::map<int32_t, int32_t> > &vec_map_nodes,
    const int32_t nb_parts
) {
    /*
     * RETURN: (description at LocalDomainStruct.h)
     * ld[x].nodes
     * ld[x].node_loctoglob
     *
     * RETURN ALSO: (description at partitioning.cpp::create_sub_domains)
     * node_is_boundary
     * vec_map_nodes
     */

    // #########################################################
    // Counting
    // #########################################################
    std::vector<int32_t> parts; // temporarily vector to store node neighboring parts ID
    std::vector<int32_t> parts_counter(nb_parts, 0); // for a fixed node `i` determine the number of neighbors for each part
    std::vector<int32_t> local_nodes_counter(nb_parts, 0); // determine the number of nodes for each parts

    //to prevent multiple allocation
    parts.reserve(100);

    for (int32_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        for (int32_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t neighbor_cell = sub_node_cellid.get(j);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);

            if (parts_counter[neighbor_part] == 0) {
                //push part ID only once
                parts.push_back(neighbor_part);
            }

            //increment neighbor part by 1 for this node
            parts_counter[neighbor_part]++;
        }
        for (int32_t j = 0; j < parts.size(); j++) {
            const int32_t part = parts[j];

            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;

            // reset parts_counter
            parts[j] = 0;
            parts_counter[part] = 0;
        }

        // reset the size of the vector
        parts.clear();
    }

    // #########################################################
    // Allocating
    // #########################################################
    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_nodes = local_nodes_counter[i];

        ld[i].nodes = new PyArray<fdx_t, 2>(make_npy_dims(nb_nodes, 3), true);
        ld[i].node_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(nb_nodes));

    }

    // #########################################################
    // Filling
    // #########################################################
    std::fill(parts_counter.begin(), parts_counter.end(), 0);
    std::fill(local_nodes_counter.begin(), local_nodes_counter.end(), 0);
    for (int32_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        const auto sub_nodes = nodes->sub_array(i);

        for (int32_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t neighbor_cell = sub_node_cellid.get(j);
            const int32_t neighbor_part = part_vert->get(neighbor_cell);
            const int32_t local_nodeid = local_nodes_counter[neighbor_part];

            if (parts_counter[neighbor_part] == 0) {
                //push part ID only once
                parts.push_back(neighbor_part);

                //*** assign node_loctoglob and vec_map_nodes
                ld[neighbor_part].node_loctoglob->get(local_nodeid) = i;
                vec_map_nodes[i][neighbor_part] = local_nodeid;
            }

            //increment neighbor part by 1 for this node
            parts_counter[neighbor_part]++;


            //*** assign nodes
            const auto sub_l_nodes = ld[neighbor_part].nodes->sub_array(local_nodeid);
            for (int32_t k = 0; k < nodes->shape[1]; k++) {
                sub_l_nodes.get(k) = sub_nodes.get(k);
            }
        }

        for (int32_t j = 0; j < parts.size(); j++) {
            const int32_t part = parts[j];
            if (parts.size() > 1) {
                // this is a boundary node

                //*** assign nodes_is_boundary
                node_is_boundary[i] = true;


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

static void loop_through_physical_faces(
    LocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    std::vector<int32_t> &vec_node_oldname,
    std::vector<int32_t> &part_phyid
) {

    /*
     * RETURN: (description at LocalDomainStruct.h)
     * ld[x].map_phy_faces
     * ld[x].max_phy_face_nodeid
     *
     * RETURN ALSO: (description at partitioning.cpp::create_sub_domains)
     * vec_node_oldname
     * part_phyid
     */

    std::vector<int32_t> intersect_cell(2);
    int total_nb_phyfaces = 0;
    for (idx_t i = 0; i < phy_faces->shape[0]; i++) {
        auto phy_face = phy_faces->sub_array(i);
        const idx_t name = phy_faces_name->get(i);
        const idx_t size = phy_face.last();
        intersect_arr(node_cellid, &phy_face, size, intersect_cell); //get the cell attached to the physical face
        //a face can be attached at most to two cells, a physical face in the other hand is attached only to one cell
        if (intersect_cell[0] != -1) {
            const int32_t cell_id = intersect_cell[0];
            const int32_t p = part_vert->get(cell_id);

            //*** local_max_phy_face_nodeid
            ld[p].max_phy_face_nodeid = std::max(size, ld[p].max_phy_face_nodeid);

            //*** local_max_phy_face_nodeid
            auto &tmp_map = ld[p].map_phyid;
            tmp_map[i] = (int32_t)tmp_map.size();

            //*** part_phyid
            part_phyid[total_nb_phyfaces] = p;
            total_nb_phyfaces++;
        }
        for (int32_t j = 0; j < size; j++) {
            const int32_t nodeid = phy_face.get(j);
            //*** vec_node_oldname
            if (vec_node_oldname[nodeid] == 0 or vec_node_oldname[nodeid] > name)
                vec_node_oldname[nodeid] = name;
        }
    }
    if (total_nb_phyfaces != phy_faces->shape[0]) {
        throw std::runtime_error("Bad input mesh, One of the physical faces is not attached to any domain cell.");
    }
}

static void loop_through_cells(
LocalDomainStruct *ld,
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
    /*
     * RETURN: (description at LocalDomainStruct.h)
     * ld[x].cells
     * ld[x].cells_type
     * ld[x].cell_loctoglob
     * ld[x].max_cell_nodeid
     * ld[x].max_cell_faceid
     * ld[x].max_face_nodeid
     * ld[x].max_cell_halonid
     * ld[x].max_halo_cell_nodeid
     * ld[x].max_node_halophyid
     * ld[x].map_int_halos
     * ld[x].map_phyid_recv
     * ld[x].map_node_halophyid
     * ld[x].map_cell_halophyid
     * ld[x].vec_node_halos
     * ld[x].max_node_phyid
     * ld[x].max_cell_phyid
     */

    std::vector<int32_t> local_nb_cells(nb_parts, 0); // for every local domain count the number of local cells
    std::vector<int32_t> visited_phyid(part_phyid.size(), -1); // used to count max_cell_phyid

    // #########################################################
    // Counting
    // #########################################################
    for (int32_t i = 0; i < cells->shape[0]; i++) {
        const int32_t part = part_vert->get(i);
        const int8_t cell_type = cells_type->get(i);
        const auto max_info = get_max_info(cell_type);

        //*** max_cell_faceid, max_face_nodeid, max_cell_nodeid
        ld[part].max_cell_faceid = std::max(max_info[0], ld[part].max_cell_faceid);
        ld[part].max_face_nodeid = std::max(max_info[1], ld[part].max_face_nodeid);
        ld[part].max_cell_nodeid = std::max(max_info[2], ld[part].max_cell_nodeid);
        local_nb_cells[part]++;
    }

    // #########################################################
    // Allocating
    // #########################################################
    for (int32_t i = 0; i < nb_parts; i++) {
        const int32_t nb_cells = local_nb_cells[i];
        const int32_t max_cell_nodeid = ld[i].max_cell_nodeid;

        ld[i].cells = new PyArray<int32_t, 2>(make_npy_dims(nb_cells, max_cell_nodeid + 1));
        ld[i].cells_type = new PyArray<int8_t, 1>(make_npy_dims(nb_cells));
        ld[i].cell_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(nb_cells));

    }

    // #########################################################
    // Filling
    // #########################################################
    std::vector<int32_t> i_visited(cells->shape[0], -1);
    std::fill(local_nb_cells.begin(), local_nb_cells.end(), 0);
    for (int32_t g_id = 0; g_id < cells->shape[0]; g_id++) {
        const int32_t part = part_vert->get(g_id);
        const int32_t l_id = local_nb_cells[part];
        const int8_t cell_type = cells_type->get(g_id);
        auto &map_int_halos = ld[part].map_int_halos;

        //*** assign local_cells_type
        ld[part].cells_type->get(l_id) = cell_type;

        //*** assign l_cell_loctoglob
        ld[part].cell_loctoglob->get(l_id) = g_id;


        const auto sub_cells = cells->sub_array(g_id);
        auto sub_l_cells = ld[part].cells->sub_array(l_id);
        int32_t nb_cell_halonid = 0;
        int32_t nb_cell_phyid = 0;
        for (int32_t j = 0; j < sub_cells.last(); j++) {
            const int32_t nodeid = sub_cells.get(j);
            const int32_t local_nodeid = vec_map_nodes[nodeid].at(part);

            //*** assign cells
            sub_l_cells.get(j) = local_nodeid;

            //check for node->cellnid only if node is boundary node
            if (node_is_boundary[nodeid]) {
                auto sub_node_cellid = node_cellid->sub_array(nodeid);
                for (int32_t k = 0; k < sub_node_cellid.last(); k++) {
                    const int32_t neighbor_cell = sub_node_cellid.get(k);
                    const int32_t neighbor_part = part_vert->get(neighbor_cell);
                    const int32_t nb_neighbor_cell_nodeid = cells->last2(neighbor_cell);

                    //condition to get cell->halo_cellnid only once
                    if (neighbor_part != part and i_visited[neighbor_cell] != g_id) {
                        i_visited[neighbor_cell] = g_id;
                        nb_cell_halonid++;

                        //*** start map_int_halos
                        //if (map_int_halos.find(neighbor_part) == map_int_halos.end()) {
                          //  map_int_halos[neighbor_part] = std::vector<int32_t>();
                            // map_int_halos[neighbor_part].reserve(1000);
                        //}
                        auto &vec_int_halos = map_int_halos[neighbor_part];
                        if (vec_int_halos.empty() or vec_int_halos.back() != l_id) {
                            vec_int_halos.push_back(l_id);
                        }
                        //*** end map_int_halos


                    }


                    if (neighbor_part != part) {
                        //*** vec_node_halos
                        const int32_t l_nodeid = vec_map_nodes[nodeid].at(neighbor_part);
                        auto &vec_node_halos = ld[neighbor_part].vec_node_halos;
                        const size_t size = vec_node_halos.size();
                        if (size == 0 or vec_node_halos[size - 1] != g_id or vec_node_halos[size - 2] != l_nodeid) { // to prevent duplication
                            vec_node_halos.push_back(l_nodeid); // local_nodeid
                            vec_node_halos.push_back(g_id); // global id of the cell
                        }
                        //*** end vec_node_halos

                        //*** max_halo_cell_nodeid
                        ld[part].max_halo_cell_nodeid = std::max(ld[part].max_halo_cell_nodeid, nb_neighbor_cell_nodeid);
                    }
                }
            }

            //*** start map_phyid_recv, map_node_halophyid
            auto sub_node_phyid = node_phyid->sub_array(nodeid);
            int32_t nb_node_phyid = 0;

            for (int32_t k = 0; k < sub_node_phyid.last(); k++) {
                const int32_t phy_id = sub_node_phyid.get(k);
                const int32_t phy_id_part = part_phyid[phy_id];

                // This is deliberately placed inside loop_through_cells and not loop_through_nodes because a node can belong to two subdomains, and part_phyid is initialized after loop_through_nodes.
                // It is ok to use map and set here the access is rare, does not affect performance.
                if (part != phy_id_part) {
                    ld[part].map_phyid_recv[phy_id_part].insert(phy_id);
                    ld[phy_id_part].map_phyid_recv[part];//create a neighborship, this trait the case when this partition when this partition only receive and does not send to a neighbor.
                    ld[part].map_node_halophyid[local_nodeid].insert(phy_id);
                    ld[part].map_cell_halophyid[l_id].insert(phy_id);
                } else {
                    nb_node_phyid++;
                    if (visited_phyid[phy_id] != g_id) {
                        visited_phyid[phy_id] = g_id;
                        nb_cell_phyid++;
                    }
                }
            }
            ld[part].max_node_phyid = std::max(ld[part].max_node_phyid, nb_node_phyid);
            //*** End map_phyid_recv, map_node_halophyid
        }
        ld[part].max_cell_phyid = std::max(ld[part].max_cell_phyid, nb_cell_phyid);
        sub_l_cells.last() = sub_cells.last();

        //*** assign max_cell_halonid
        ld[part].max_cell_halonid = std::max(ld[part].max_cell_halonid, nb_cell_halonid);

        local_nb_cells[part]++;
    }
}

static PyObject *get_result_as_py_list(LocalDomainStruct *ld, const int32_t nb_parts) {
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

PyObject *create_sub_domains(
    LocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<fdx_t, 2> *nodes,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    PyArray<int32_t, 2> *node_phyid,
    const int32_t nb_parts,
    const int32_t dim
    ) {


    std::vector<int32_t> vec_node_oldname(nodes->shape[0]); // store the node oldname
    std::vector<int32_t> part_phyid(phy_faces->shape[0]); // store the physical face partition ID
    std::vector<bool> node_is_boundary(nodes->shape[0], false); // for every g_node assign True if g_node has neighboring cells form different parts.
    std::vector<std::map<int32_t, int32_t> > vec_map_nodes(nodes->shape[0]); // for every g_node store local id of a specific partition
    std::vector<int32_t> vec_cell_to_halo(cells->shape[0], -1); // cells to haloext_id used by `create_halos` for `ld[p].node_halos`
    
    //part1
    DEBUG_TIME_IT("");
    loop_through_nodes(ld, part_vert, node_cellid, nodes, node_is_boundary, vec_map_nodes, nb_parts);
    loop_through_physical_faces(ld, part_vert, node_cellid, phy_faces, phy_faces_name, vec_node_oldname, part_phyid);
    loop_through_cells(ld, part_vert, node_cellid, cells, cells_type, node_phyid, node_is_boundary, vec_map_nodes, part_phyid, nb_parts);
    DEBUG_TIME_IT("loop_through");
    //part2
    DEBUG_TIME_IT("");
    for (int32_t p = 0; p < nb_parts; p++) {
        create_halos(ld, cells, nodes, vec_cell_to_halo, p, dim);
        create_phy(ld, p, phy_faces_name, phy_faces, vec_node_oldname, vec_map_nodes);
    }
    DEBUG_TIME_IT("create_halos, create_phy");

    return get_result_as_py_list(ld, nb_parts);
}
