//
// Created by aben-ham on 9/22/25.
// Last modification on 3/27/26.
//

#ifndef LOCALDOMAINSTRUCT_H
#define LOCALDOMAINSTRUCT_H

#include "PyArray.h"
#include <map>
#include <set>
#include <vector>
#include "Types.h"
#include <metis.h>

/**
 * @brief Represents a single partitioned subdomain (local domain) of a global computational mesh.
 *
 * This structure holds all the local mesh geometry/topology, global-to-local
 * mapping vectors, halo, ghost exchange structures for MPI communication, and
 * physical boundary condition vectors. It heavily utilizes `PyArray` for NumPy C-API.
 */
struct LocalDomainStruct {
    LocalDomainStruct(const LocalDomainStruct&) = delete;                   // delete copy constructor
    LocalDomainStruct& operator=(const LocalDomainStruct&) = delete;        // delete copy assignment
    LocalDomainStruct()= default;
    ~LocalDomainStruct();

    // =========================================================================
    // Core Mesh Topology
    // =========================================================================
    /// @brief Local node coordinates. [nb_nodes, 3] layout: [[x, y, z], ...]
    PyArray<fdx_t, 2> *nodes = nullptr;

    /// @brief Cell connectivity. [nb_cells, max_cell_nodeid+1] layout: [[node1, node2, ..., nb_nodes], ...]
    PyArray<idx_t, 2> *cells = nullptr;

    /// @brief Geometric types of each cell. [nb_cells]. look at (CELL_TYPE at manapy_part.h)
    PyArray<int8_t, 1> *cells_type = nullptr;

    /// @brief Physical faces connectivity. [nb_phy_faces, max_phy_face_nodeid + 1] layout: [[node1, node2, ..., nb_nodes], ...]
    PyArray<idx_t, 2> *phy_faces = nullptr;

    /// @brief Physical boundary names. [nb_phy_faces]
    PyArray<idx_t, 1> *phy_faces_name = nullptr;

    // =========================================================================
    // Global/Local Mappings
    // =========================================================================

    /// @brief Maps local cell IDs back to global cell IDs. [nb_cells]
    PyArray<idx_t, 1> *cell_loctoglob = nullptr;

    /// @brief Maps local node IDs back to global node IDs. [nb_nodes]
    PyArray<idx_t, 1> *node_loctoglob = nullptr;

    /// @brief Original node boundary names before partitioning. [nb_nodes]
    PyArray<idx_t, 1> *node_oldname = nullptr;


    // =========================================================================
    // Halo Exchange Structures (Parallel Communication)
    // =========================================================================

    /// @brief Halo neighbor subdomains. [2, nb_neighbor_parts] layout: [[NeighborP1ID, ...], [NbHalosIntConnectedToP1, ...]]
    PyArray<idx_t, 2> *halo_neighsub = nullptr;

    /// @brief Connects local boundary nodes to exterior halo cells. shape=(2 * nb_halos); pairs: (LocalNodeID, HaloID) HaloId is an index point to halo_halosext
    PyArray<idx_t, 1> *node_halos = nullptr;

    /// @brief Exterior halos cells (cells received from neighbors). shape=(nb_halos, max_cell_nodeid + 2) layout:[[GlobalId, Node1, Node2..., NbNodes], ...]
    /// @details halo_halosext is constructed by concatenating halo_halosint cells of the neighboring part (!important) using the order of neighboring partition in halo_neighsub
    PyArray<idx_t, 2> *halo_halosext = nullptr;

    /// @brief These are local cells inside *this* partition that need to be packaged and sent off to neighboring partitions to serve as *their* exterior halos. Contains local cell IDs, grouped by receiving neighbor partition. [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    /// @details The order and number of HalosIntConnectedToPX is specified in `halo_neighsub` as `halo_neighsub[PX][0] = PartitionId` and `halo_neighsub[PX][1] = NbHalosIntConnectedToPX`
    PyArray<idx_t, 1> *halo_halosint = nullptr;

    /// @brief Centroids/Volumes of exterior halos. [nb_halos, 4] layout: [[center_x, center_y, center_z, volume/area], ...]
    PyArray<fdx_t, 2> *halo_centvol = nullptr;

    // =========================================================================
    // Phyid Communication
    // =========================================================================


    /// @brief Store neighbor partition ID; number of phyids sent from this part. number of phyids received from this neighbor;. [nb_neighbor_parts, 3] layout [[Neighbor partition ID, nb_send, nb_recv] ...]
    PyArray<idx_t, 2> *phyid_neighbor = nullptr;

    /// @brief Exterior phyids received from neighbour partitions,
    ///        ordered exactly as they appear in `phyid_neighbor`.
    ///        [nb_halo_ghost aka exterior phyid] layout [globalId...]
    PyArray<idx_t, 1> *phyid_recv = nullptr;

    /// @brief Indices poit to phy_faces table that represent local physical faces that have to be packaged and sent off to neighbouring
    ///        partitions so that they become the *exterior* phyids of those neighbours.
    ///        layout [indices poit to phy_faces table for the first neighbor partition, indices poit to phy_faces table for the second neighbor partition ...]
    ///
    /// @details The order of these ids follows the same neighbour order as in
    ///          `phyid_neighbor`.
    PyArray<idx_t, 1> *phyid_send = nullptr;

    /// @brief Links local nodes to exterior physical boundary conditions. layout: [NodeLocalId1, Size1, Size1 IndicesPointToTheTargetPhyId_recv..., NodeLocalId2, Size2, Size2 Indices..., ...., SizeN]
    PyArray<idx_t, 1> *node_halophyid = nullptr;

    /// @brief Links local cells to exterior physical boundary conditions. layout: [CellLocalId1, Size1, Size1 IndicesPointToTheTargetPhyId_recv..., CellLocalId2, Size2, Size2 Indices..., ...., SizeN]
    PyArray<idx_t, 1> *cell_halophyid = nullptr;

    PyObject *tuple_res = nullptr;

    // =========================================================================
    // Scalars (Pre-calculated dimensional sizes)
    // =========================================================================
    idx_t max_cell_nodeid = 0;       ///< Max number of nodes in any local cell
    idx_t max_cell_faceid = 0;       ///< Max number of faces on any local cell
    idx_t max_face_nodeid = 0;       ///< Max number of nodes on any local face
    idx_t max_node_haloid = 0;       ///< Max neighbor halo cells connected across any local node
    idx_t max_cell_halonid = 0;      ///< Max neighbor halo cells connected across any local cell
    idx_t max_halo_cell_nodeid = 0;  ///< Max node count for any exterior halo cell
    idx_t max_node_phyid = 0;        ///< Max connected boundary physical faces count for any local node
    idx_t max_node_halophyid = 0;    ///< Max connected exterior boundary physical faces count for any local node
    idx_t max_cell_phyid = 0;        ///< Max connected boundary physical faces count for any local cell
    idx_t max_cell_halophyid = 0;    ///< Max connected exterior boundary physical faces count for any local cell


    // =========================================================================
    // Temporary Construction Variables
    // =========================================================================
    std::map<idx_t, std::vector<idx_t> > map_int_halos; ///< Neighbor partitionId => List of interior halos by local cellID
    std::vector<idx_t> vec_node_halos;            ///< Vector equivalent to `node_halos` storing global cell IDs directly. it stores the Pair(LocalNodeID, GlobalCellId)
    idx_t max_phy_face_nodeid = 0;                    ///< Max nodes on any local physical face
    std::map<idx_t, idx_t> map_phyid;               ///< Global ID of phy_face => Local ID of phy_face
    std::map<idx_t, std::set<idx_t> > map_phyid_recv; ///< PartitionId => Set(global id of exterior phyids) represent boundary physical faces that will be received from a neighbor Partition, Important if this partition does not receive from and only send to a partition, an empty set inserted to represent neighborship.
    std::map<idx_t, std::set<idx_t> > map_node_halophyid; ///< GlobalNodeId => Set(global id of exterior phyids)
    std::map<idx_t, std::set<idx_t> > map_cell_halophyid; ///< GlobalCellId=> Set(global id of exterior phyids)




public:
    /**
     * @brief Packs Python ownership of the assigned NumPy array pointers into a Python tuple.
     */
    void create_tuple(); // return a python tuple to return the data latter

private:

    /**
     * @brief frees memory for any `PyArray` tables.
     */
    void    free_tables();

};




#endif //LOCALDOMAINSTRUCT_H
