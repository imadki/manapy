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

    /// @brief Links local nodes to exterior physical boundary conditions. shape=(nb_nodes, max_node_halophyid + 1) layout: [[LocalPhysicalId1, LocalPhysicalId2, ..., NbConnectedPhysicalId], ...]
    /// @details LocalPhysicalId1 represent the localId of physical boundary conditions at the *neighbor* part as the index can't be mapped in *this* partition.
    PyArray<int32_t, 2> *node_halophyid = nullptr;

    PyArray<int32_t, 1> *phyid_recv = nullptr; // int32[:] [boundary faces global index, ...] description="store physical faces of this partition by its local index and for the other partitions by global index, all other tables that will use boundary faces must point to this table"
    PyArray<int32_t, 1> *phyid_recv_part_size = nullptr; // int32[:] [boundary faces partId, size]
    PyArray<int32_t, 1> *phyid_send = nullptr;  // int32[:] self.phyid_send = np.zeros(1, dtype=np.int32) # [recv_part_index, size, `size` indices point to phyid_recv, ...] description="used when this part need to send its boundary faces to recv_part"
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



    // =========================================================================
    // Temporary Construction Variables
    // Used exclusively during the domain allocation phases; destroyed afterward.
    // =========================================================================
    std::map<int, std::vector<int> > map_int_halos; ///< Neighbor partitionId => List of interior halos by local cellID
    int max_node_halophyid = 0;                     ///< Max exterior physical IDs on a node
    int max_phy_face_nodeid = 0;                    ///< Max nodes on a physical face
    std::map<int, int> map_phy_faces;               ///< Global ID of phy_face => Local ID of phy_face
    std::set<int> set_phyids;                       ///< Set of all global Ids of phyids required by this partition (local + exterior)
    std::set<int> set_halo_phyid_neighsub;          ///< Set of partition IDs corresponding to exterior phyids
    std::vector<int32_t> vec_phyids;                ///< Sorted version of `set_phyids` by partitionId of the physical face
    std::map<int32_t, int32_t> map_phyids;          ///< Global phyid => Index inside `vec_phyids`
    std::vector<int32_t> vec_node_halos;            ///< Vector equivalent to `node_halos` storing global cell IDs directly (LocalNodeID, GlobalCellId)




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
