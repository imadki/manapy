#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"
#include "LocalDomainStruct.h"

/*
 * local_node_cellid
 * local_b_node
 * nb_node_halos => node_cellid - local_node_cellid
 */

/*
 * nodes
 * cells
 * cells_type
 * cell_loctoglob
 * node_loctoglob
 * @node_halos
 * @halo_neighsub
 * @halo_halosext
 * @halo_halosint
 * @max_node_haloid,
 * max_cell_nodeid,
 * max_cell_faceid,
 * max_face_nodeid,
 * max_cell_halonid
 */

// #################################################################
// 1. _create_sub_domains
// #################################################################

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


    const int32_t nb_parts = 10;
    std::vector<int32_t> local_nb_nodes(nb_parts, 0);
    std::vector<int32_t> tmp(nb_parts, -1);
    std::vector<int32_t> max_cellid(nb_parts, 0);
    for (idx_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        for (idx_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t n_cellid = sub_node_cellid.get(j);
            const int32_t n_cellid_part = part_vert->get(n_cellid);
            if (tmp[n_cellid_part] == -1 || tmp[n_cellid_part] != i) {
                local_nb_nodes[n_cellid_part]++;
                tmp[n_cellid_part] = i;
            }
        }
    }

    std::vector<PyArray<int32_t, 2> *> local_node_cellids(nb_parts, nullptr);
    for (idx_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        for (idx_t j = 0; j < sub_node_cellid.last(); j++) {
            const int32_t n_cellid = sub_node_cellid.get(j);
            const int32_t n_cellid_part = part_vert->get(n_cellid);
            if (local_node_cellids[n_cellid_part] == nullptr) {
                const int32_t nb_nodes = local_nb_nodes[i];
                const int32_t max_n_cellid = max_cellid[i];
                local_node_cellids[n_cellid_part] = new PyArray<int32_t, 2>(make_npy_dims(nb_nodes, max_n_cellid));
            }

            auto local_node = local_node_cellids[n_cellid_part]->sub_array(i);
            local_node.get(counter) = n_cellid;
            if (tmp[n_cellid_part] == -1 || tmp[n_cellid_part] != i) {
                local_nb_nodes[n_cellid_part]++;
                tmp[n_cellid_part] = i;
            }
        }
    }




}






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