//
// Created by aben-ham on 9/22/25.
// Last modification on 3/12/26.
//

#ifndef LOCALDOMAINSTRUCT_H
#define LOCALDOMAINSTRUCT_H

#include "PyArray.h"
#include <map>
#include <set>
#include <vector>
#include "Types.h"

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
    PyArray<int32_t, 2> *cells = nullptr;

    /// @brief Geometric types of each cell. [nb_cells]. look at (CELL_TYPE)
    PyArray<int8_t, 1> *cells_type = nullptr;

    /// @brief Physical faces connectivity. [nb_phy_faces, max_phy_face_nodeid + 1] layout: [[node1, node2, ..., nb_nodes], ...]
    PyArray<int32_t, 2> *phy_faces = nullptr;

    /// @brief Physical boundary names. [nb_phy_faces]
    PyArray<int32_t, 1> *phy_faces_name = nullptr;

    // =========================================================================
    // Global/Local Mappings
    // =========================================================================

    /// @brief Maps local cell IDs back to global cell IDs. [nb_cells]
    PyArray<int32_t, 1> *cell_loctoglob = nullptr;

    /// @brief Maps local node IDs back to global node IDs. [nb_nodes]
    PyArray<int32_t, 1> *node_loctoglob = nullptr;

    /// @brief Original node boundary names before partitioning. [nb_nodes]
    PyArray<int32_t, 1> *node_oldname = nullptr;


    // =========================================================================
    // Halo Exchange Structures (Parallel Communication)
    // =========================================================================

    /// @brief Halo neighbor subdomains. [2, nb_neighbor_parts] layout: [[NeighborP1ID, ...], [NbHalosIntConnectedToP1, ...]]
    PyArray<int32_t, 2> *halo_neighsub = nullptr;

    /// @brief Connects local boundary nodes to exterior halo cells. shape=(2 * nb_halos); pairs: (LocalNodeID, HaloID) HaloId is an index point to halo_halosext
    PyArray<int32_t, 1> *node_halos = nullptr;

    /// @brief Exterior halos cells (cells received from neighbors). shape=(nb_halos, max_cell_nodeid + 2) layout:[[GlobalId, Node1, Node2..., NbNodes], ...]
    /// @details halo_halosext is constructed by concatenating halo_halosint cells of the neighboring part (!important) using the order of neighboring partition in halo_neighsub
    PyArray<int32_t, 2> *halo_halosext = nullptr;

    /// @brief These are local cells inside *this* partition that need to be packaged and sent off to neighboring partitions to serve as *their* exterior halos. Contains local cell IDs, grouped by receiving neighbor partition. [HalosIntConnectedToP1 halos ..., HalosIntConnectedToP2 halos ..., ...]
    /// @details HalosIntConnectedToPX... represent the local ID of the cell located in PX
    /// @details The order and number of HalosIntConnectedToPX is specified in `halo_neighsub` as `halo_neighsub[PX][0] = PartitionId` and `halo_neighsub[PX][1] = NbHalosIntConnectedToPX`
    PyArray<int32_t, 1> *halo_halosint = nullptr;

    /// @brief Centroids/Volumes of exterior halos. [nb_halos, 4] layout: [[center_x, center_y, center_z, volume/area], ...]
    PyArray<fdx_t, 2> *halo_centvol = nullptr;

    // =========================================================================
    // Phyid Communication
    // =========================================================================


    /// @brief Store neighbor partition ID; number of phyids received from this neighbor; number of phyids sent from this part. [[Neighbor partition ID, nb_send, nb_recv] ...]
    PyArray<int32_t, 2> *phyid_neighbor = nullptr;

    /// @brief Exterior phyid by globalId (phyid receuved from neighbors), fallowing the same order of neighbors as in phyid_neighbor
    PyArray<int32_t, 1> *phyid_recv = nullptr;

    /// @brief These are local phyids inside *this* partition that need to be packaged and sent off to neighboring partitions to serve as *their* exterior phyids. Store phyids by it localId for every partition.
    /// @details The order of neighbors matches that in `phyid_neighbor` [LocalPhyIdOfP0, ... LocalPhyIdOfPn]
    PyArray<int32_t, 1> *phyid_send = nullptr;

    /// @brief Links local nodes to exterior physical boundary conditions. layout: [NodeLocalId1, IndexPointToPhyId_recv, ... Size1, NodeLocalId2, ... Size2, ...., SizeN]
    /// @details IndexPointToPhyId_recv point to phyid_recv.
    PyArray<int32_t, 1> *node_halophyid = nullptr;

    PyArray<int32_t, 1> *cell_halophyid = nullptr;

    PyObject *tuple_res = nullptr;

    // =========================================================================
    // Scalars (Pre-calculated dimensional sizes)
    // =========================================================================
    int max_cell_nodeid = 0;       ///< Max number of nodes in any cell
    int max_cell_faceid = 0;       ///< Max number of faces on any cell
    int max_face_nodeid = 0;       ///< Max number of nodes on any face
    int max_node_haloid = 0;       ///< Max neighbor halo cells connected across any single node
    int max_cell_halonid = 0;      ///< Max neighbor halo cells connected across any single cell
    int max_halo_cell_nodeid = 0;  ///< Max node count for any exterior halo cell
    int max_node_phyid = 0;
    int max_node_halophyid = 0;
    int max_cell_phyid = 0;
    int max_cell_halophyid = 0;


    // =========================================================================
    // Temporary Construction Variables
    // =========================================================================
    std::map<int32_t, std::vector<int32_t> > map_int_halos; ///< Neighbor partitionId => List of interior halos by local cellID
    std::vector<int32_t> vec_node_halos;            ///< Vector equivalent to `node_halos` storing global cell IDs directly (LocalNodeID, GlobalCellId)
    int32_t max_phy_face_nodeid = 0;                    ///< Max nodes on any physical face
    std::map<int32_t, int32_t> map_phyid;               ///< Global ID of phy_face => Local ID of phy_face
    std::map<int32_t, std::set<int32_t> > map_phyid_recv; ///< PartitionId => Set(global id of exterior phyids)
    std::map<int32_t, std::set<int32_t> > map_node_halophyid; ///< GlobalNodeId => Set(global id of exterior phyids)
    std::map<int32_t, std::set<int32_t> > map_cell_halophyid;




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
