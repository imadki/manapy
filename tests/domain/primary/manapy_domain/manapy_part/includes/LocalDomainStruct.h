//
// Created by aben-ham on 9/22/25.
//

#ifndef LOCALDOMAINSTRUCT_H
#define LOCALDOMAINSTRUCT_H

#include "PyArray.h"
#include <map>
#include <set>
#include <vector>
#include "Types.h"

//TODO comments

struct LocalDomainStruct {
    LocalDomainStruct(const LocalDomainStruct&) = delete;                   // delete copy constructor
    LocalDomainStruct& operator=(const LocalDomainStruct&) = delete;        // delete copy assignment
    LocalDomainStruct()= default;


    PyArray<fdx_t, 2> *nodes = nullptr; // float64 or float32[:, :] [[node x, y, z]]
    PyArray<int32_t, 2> *cells = nullptr; // int32[:, :] [[cells nodes]]
    PyArray<int8_t, 1> *cells_type = nullptr; // int8[:] [cell type]
    PyArray<int32_t, 2> *phy_faces = nullptr; // int32[:, :] [[physical face nodes]]
    PyArray<int32_t, 1> *phy_faces_name = nullptr; // int32[:] [physical face name]
    PyArray<int32_t, 1> *cell_loctoglob = nullptr; // int32[:] [cell global index]
    PyArray<int32_t, 1> *node_loctoglob = nullptr; // int32[:] [node global index]
    PyArray<int32_t, 1> *node_oldname = nullptr; // int32[:] [node old name, ...]
    PyArray<int32_t, 2> *halo_neighsub = nullptr; // int32[:, :] [[NeighborP1ID, NeighborP2ID, ...], [NbHalosIntConnectedToP1, ...]]
    PyArray<int32_t, 1> *node_halos = nullptr; // int32[:] [NodiId, haloId, ...] shape=(2 * nb_halos) couple (NodeId, haloId) for each exthalo, HaloId is an index point to halo_halosext, nodeId is the local nodeId.
    PyArray<int32_t, 2> *node_halophyid = nullptr; // int32[:, :] [[index0 point to halo_halobf, index1 ..., size]] shape=(nb_nodes, max_node_halobf + 1)
    PyArray<int32_t, 2> *halo_halosext = nullptr; // int32[:, :] [[global index of halocell, global index of cell nodes, size]] shape=(nb_halos, max_cell_nodeid + 2)
    PyArray<int32_t, 1> *halo_halosint = nullptr; // int32[:] [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    PyArray<fdx_t, 2> *halo_centvol = nullptr; // float64 or float32[:, :] [halocell_center_{x, y, z}, halocell_volume_{x, y, z}] # z axis only on 3D
    PyArray<int32_t, 1> *phyid_recv = nullptr; // int32[:] [boundary faces global index, ...] description="store physical faces of this partition by its local index and for the other partitions by global index, all other tables that will use boundary faces must point to this table"
    PyArray<int32_t, 1> *phyid_recv_part_size = nullptr; // int32[:] [boundary faces partId, size]
    PyArray<int32_t, 1> *phyid_send = nullptr;  // int32[:] self.phyid_send = np.zeros(1, dtype=np.int32) # [recv_part_index, size, `size` indices point to phyid_recv, ...] description="used when this part need to send its boundary faces to recv_part"
    PyObject *tuple_res = nullptr;

    // Scalars
    int max_cell_nodeid = 0;
    int max_cell_faceid = 0;
    int max_face_nodeid = 0;
    int max_node_haloid = 0; // max neighbor halo cells across all nodes
    int max_cell_halonid = 0; // max neighbor halo cells across all cells



    // Temporary members used to generate the above tables and scalars
    std::map<int, std::vector<int> > map_int_halos; // map(partitionId => listOf interior halos)
    int max_node_halophyid = 0;
    int max_phy_face_nodeid = 0;
    std::map<int, int> map_phy_faces; // map(g_id of phy_face => local if of phy_face)
    std::set<int> set_phyids; // set of all phyids needed by the partition either local of halos
    std::set<int> set_halo_phyid_neighsub; // set of partition ids of the halo phyids
    std::vector<int32_t> vec_phyids; // vec(this->set_phyids) sorted by partitionID of each item
    std::map<int32_t, int32_t> map_phyids; // g_phyid => g_phyid index in this->vec_phyids
    std::vector<int32_t> vec_node_halos; // the same as this->node_halos
    std::vector<int32_t> vec_halos; // list of halo cells (g_id of cell)

    ~LocalDomainStruct();


public:

    void create_tuple(); // return a python tuple to return the data latter

private:

    void    free_tables();

};




#endif //LOCALDOMAINSTRUCT_H
