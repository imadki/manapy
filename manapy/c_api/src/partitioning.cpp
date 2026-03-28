#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "manapy_part.h"

/**
 * @file partitioning.cpp
 * @brief Mesh partitioning logic: splits a global mesh into local subdomains (partitions)
 *        for distributed / MPI-based numerical solvers.
 *
 * Helper Class VecMapNodes
 *
 * Pipeline called via create_sub_domains:
 *
 *   loop_through_nodes
 *   loop_through_physical_faces
 *   loop_through_cells
 *   create_halos  (per part)
 *   create_phy    (per part)
 *   get_result_as_py_list
 *
 * Public API:
 *   create_sub_domains — entry point called from the Python C-extension layer.
 */


/// @brief A small helper that stores a mapping from *global node* IDs to the
///        pair ``(partition_id, local_node_id)`` for each partition.
///
/// - insert method for adding global_id => pair
/// - operator(node_id, partition_id) to getting the local_node_id
///
/// - Because a node is shared by at most a handful of partitions (typically 1–4), a small flat vector with linear search
/// outperforms a red-black tree (std::map)
/// - Copy/assignment constructors are deleted
class VecMapNodes {
public:
    VecMapNodes(const VecMapNodes &) = delete;
    VecMapNodes &operator=(const VecMapNodes &) = delete;
    ~VecMapNodes() = default;

private:
    using Pair = std::pair<idx_t, idx_t>;
    std::vector<std::vector<Pair>> map_nodes;

public:
    explicit VecMapNodes(const size_t size) : map_nodes(size) {
        for (auto &v : map_nodes) {
            v.reserve(4);
        }
    }

    void insert(const idx_t node_id, const idx_t partition_id, const idx_t local_node_id) {
        map_nodes[node_id].emplace_back(partition_id, local_node_id);
    }

    idx_t operator()(const idx_t node_id, const idx_t partition_id) const {
        const auto &vec = map_nodes[node_id];

        const Pair* data = vec.data();
        const Pair* end  = data + vec.size();

        for (; data != end; ++data) {
            if (data->first == partition_id) {
                return data->second;
            }
        }
        throw std::out_of_range("partition id out of range");
    }
};

// ============================================================================
// create_halos
// ============================================================================
/**
 * @brief Builds halo exchange structures for partition @p p.
 * "Halos" of partition P are the cells that physically reside in neighbouring
 * partitions but are needed by P for computations at shared boundaries.
 * RETURN: (description at LocalDomainStruct.h)
     * ld[x].max_node_haloid
     * ld[x].node_halos
     * ld[x].halo_halosext
     * ld[x].halo_neighsub
     * ld[x].halo_halosint
     * ld[x].halo_centvol
 * @param ld              Array of all partition descriptors.
 * @param cells           Global cell-node connectivity [nb_cells, max_nodeid+1].
 * @param nodes           Global node coordinates [nb_nodes, 3].
 * @param vec_cell_to_halo Scratch: maps global cell id → exterior halo index (output).
 * @param vec_max         Scratch: reused across calls to avoid re-allocation. Must be
 *                        at least max_local_nodes in size; caller resets used entries.
 * @param p               Partition index being filled.
 * @param dim             Spatial dimension (2 or 3).
 */
static void create_halos(
    LocalDomainStruct *ld,
    PyArray<idx_t, 2> *cells,
    PyArray<fdx_t, 2> *nodes,
    std::vector<idx_t> &vec_cell_to_halo,
    std::vector<idx_t> &vec_max,// pre-allocated buffer, reset before used
    const idx_t p,
    const idx_t dim
    ) {

    // Aliases for frequently accessed read-only members of this partition.
    const auto &l_map_int_halos = ld[p].map_int_halos;
    const auto &vec_node_halos = ld[p].vec_node_halos;

    //write
    idx_t &l_max_node_haloid = ld[p].max_node_haloid;
    l_max_node_haloid = -1;

    // -------------------------------------------------------------------------
    // Return halo_halosext, node_halos, max_node_haloid
    // Populate  vec_cell_to_halo
    // -------------------------------------------------------------------------
    // The exterior halos of P are exactly the interior cells of each neighbour
    // that share a face with P (recorded in map_int_halos during loop_through_cells).
    // Global cell IDs are unique across partitions, so there are no duplicates.

    //local
    idx_t nb_halos = 0;

    //reset vec_max
    for (idx_t i = 0; i < vec_node_halos.size(); i += 2) {
        vec_max[vec_node_halos[i]] = 0;
    }

    // Count total exterior halos (= sum of interior halos over all neighbours).
    for (const auto &item: ld[p].map_int_halos) {
        const auto neighbor = item.first;
        const auto &halos_int = ld[neighbor].map_int_halos[p];
        nb_halos += static_cast<idx_t>(halos_int.size());
    }

    // Allocate halo_halosext
    ld[p].halo_halosext = new PyArray<idx_t, 2>(make_npy_dims(nb_halos, ld[p].max_halo_cell_nodeid + 2));
    const auto *l_halo_halosext = ld[p].halo_halosext;

    // Fill halo_halosext and populate vec_cell_to_halo (g_cell_id → halo row index).
    idx_t counter = 0;
    for (const auto &item: ld[p].map_int_halos) {
        const idx_t neighbor = item.first;
        const auto &halos_int = ld[neighbor].map_int_halos[p]; // interiors of the neighbor partition.

        for (const auto &halo: halos_int) {
            const idx_t g_id = ld[neighbor].cell_loctoglob->get(halo);

            //*** vec_cell_to_halo
            vec_cell_to_halo[g_id] = counter;

            //*** start halo_halosext
            //row: [g_id, node0, node1, ..., nb_nodes_in_cell]
            auto sub_l_halo_halosext = l_halo_halosext->sub_array(counter);
            sub_l_halo_halosext.get(0) = g_id;
            auto sub_cells = cells->sub_array(g_id);
            for (idx_t j = 0; j < sub_cells.last(); j++) {
                const idx_t nodeid = sub_cells.get(j);
                sub_l_halo_halosext.get(j + 1) = nodeid;
            }
            sub_l_halo_halosext.last() = sub_cells.last() + 1;
            //*** end halo_halosext

            counter++;
        }
    }

    // Allocate and fill node_halos.
    // vec_node_halos is a flat array of pairs built during loop_through_cells:
    //   even indices: local_node_id
    //   odd  indices: global_cell_id of the touching exterior halo cell
    ld[p].node_halos = new PyArray<idx_t, 1>(make_npy_dims(vec_node_halos.size()));
    const auto l_node_halos = ld[p].node_halos;

    //*** node_halos, max_node_haloid
    // Even pass: copy local_node_id and track how many halo cells touch each node.
    for (idx_t i = 0; i < vec_node_halos.size(); i=i+2) {
        const idx_t local_node_id = vec_node_halos[i];
        // copy node_id
        l_node_halos->get(i) = local_node_id;
        // determine max_node_haloid
        l_max_node_haloid = std::max(l_max_node_haloid, ++vec_max[local_node_id]);
    }
    // Odd pass: translate global_cell_id → exterior halo index using vec_cell_to_halo.
    for (idx_t i = 1; i < vec_node_halos.size(); i=i+2) {
        l_node_halos->get(i) = vec_cell_to_halo[vec_node_halos[i]];
    }


    // -------------------------------------------------------------------------
    // halo_neighsub, halo_halosint
    // -------------------------------------------------------------------------
    // halo_neighsub[0][k] = partition_id of k-th neighbour
    // halo_neighsub[1][k] = number of interior-halo cells this partition sends to neighbour k
    // halo_halosint        = flat list of local cell IDs to send, grouped by neighbour

    idx_t nb_halos_int = 0;
    for (const auto &item : l_map_int_halos) {
        nb_halos_int += (idx_t)item.second.size();
    }

    ld[p].halo_neighsub = new PyArray<idx_t, 2>(make_npy_dims(2, l_map_int_halos.size()));
    ld[p].halo_halosint = new PyArray<idx_t, 1>(make_npy_dims(nb_halos_int));
    const auto *l_halo_neighsub = ld[p].halo_neighsub;
    const auto *l_halo_halosint = ld[p].halo_halosint;

    idx_t neighsub_counter = 0;
    idx_t halosint_counter = 0;
    for (const auto &item : l_map_int_halos) {
        const idx_t partition = item.first;
        const auto &set = item.second;// local cell IDs of interior halos for this neighbour

        //***halo_neighsub
        l_halo_neighsub->sub_array(0).get(neighsub_counter) = partition;
        l_halo_neighsub->sub_array(1).get(neighsub_counter) = (idx_t)set.size();
        neighsub_counter += 1;
        for (const auto interior_cell: set) {
            //*** halo_halosint
            l_halo_halosint->get(halosint_counter) = interior_cell;
            halosint_counter += 1;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3 – halo_centvol
    // -------------------------------------------------------------------------
    // Compute and store geometric centroid (x,y,z) + volume (or area in 2D)
    // for every exterior halo cell.
    //*** start halo_centvol
    ld[p].halo_centvol = new PyArray<fdx_t, 2>(make_npy_dims(nb_halos, 4)); // x, y, z, vol/area
    if (dim == 2)
        compute_halo_cell_center_area_2d(ld[p].halo_halosext, nodes, ld[p].halo_centvol);
    else if (dim == 3)
        compute_halo_cell_center_volume_3d(ld[p].halo_halosext, nodes, ld[p].halo_centvol);
    //*** end halo_centvol
}

// ============================================================================
// create_phy
// ============================================================================
/**
 * @brief Builds the physical boundary communication arrays for partition @p p.
 *
 * Physical faces that straddle a partition boundary must be communicated between
 * the partition that owns the face (interior side) and its neighbour (which sees
 * the face as an exterior / halo physical face).  This function produces:
 *   * RETURN: (description at LocalDomainStruct.h)
 *   - ld[x].phyid_neighbor  — [nb_neighbors, 3]: (neighbour_id, nb_send, nb_recv) per neighbour.
 *   - ld[x].phyid_recv      — flat list of exterior physical face global IDs to receive grouped by neighbours.
 *   - ld[x].phyid_send      — flat list of local physical face IDs to send to neighbours grouped by neighbours.
 *   - ld[x].node_halophyid  — per-node list of exterior phyid indices (into phyid_recv).
 *   - ld[x].cell_halophyid  — per-cell list of exterior phyid indices (into phyid_recv).
 *   - ld[x].node_oldname    — for each local node: original boundary name from the mesh before partitioning.
 *   - ld[x].phy_faces       — local physical face connectivity with local node IDs.
 *   - ld[x].phy_faces_name  — names (tags) of local physical faces.
 *   - ld[x].max_node_halophyid
 *   - ld[x].max_cell_halophyid
 *
 * @param ld              Array of all partition descriptors.
 * @param p               Partition index being filled.
 * @param phy_faces_name  Global physical face name tags [nb_phy_faces].
 * @param phy_faces       Global physical face connectivity [nb_phy_faces, max_nodeid+1].
 * @param vec_node_oldname Original boundary names indexed by global node ID.
 * @param vec_map_nodes   Global-to-local node ID map (all partitions).
 */
static void create_phy(
    LocalDomainStruct *ld,
    const idx_t p,
    const PyArray<idx_t, 1> *phy_faces_name,
    const PyArray<idx_t, 2> *phy_faces,
    const std::vector<idx_t> &vec_node_oldname,
    const VecMapNodes &vec_map_nodes
    ) {

    // Aliases for read-only data of this partition.
    // Description in the header file
    const auto &l_map_phy_faces = ld[p].map_phyid;
    const auto max_phy_face_nodeid = ld[p].max_phy_face_nodeid;
    const auto l_node_loctoglob = ld[p].node_loctoglob;
    const auto l_nb_nodes = static_cast<idx_t>(l_node_loctoglob->shape[0]);
    const auto &l_map_phyid_recv = ld[p].map_phyid_recv;
    const auto &l_map_node_halophyid = ld[p].map_node_halophyid;
    const auto &l_map_cell_halophyid = ld[p].map_cell_halophyid;

    /// Temporary map: exterior phyid global id → offset inside phyid_recv (used to fill node/cell_halophyid).
    std::map<idx_t, idx_t> map_halophyid; ///< Map halophyid to its location inside phyid_recv

    // -------------------------------------------------------------------------
    // Return phyid_neighbor phyid_recv phyid_send node_halophyid cell_halophyid, max_node_halophyid, max_cell_halophyid
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    // Compute sizes for all output arrays
    // -------------------------------------------------------------------------
    idx_t neighbor_size       = 0;
    idx_t recv_size           = 0;
    idx_t send_size           = 0;
    idx_t node_halophyid_size = 0;
    idx_t cell_halophyid_size = 0;

    //*** Compute sizes for phyid_neighbor phyid_recv phyid_send node_halophyid
    for (const auto &[neighbor_part_id, set_halophyid]: l_map_phyid_recv) {
        recv_size += static_cast<idx_t>(set_halophyid.size());
        // The number of phyids to send equals those the neighbour receives from us.
        send_size += static_cast<idx_t>(ld[neighbor_part_id].map_phyid_recv[p].size());
        neighbor_size++;
    }
    for (const auto& [_, set_node_halophyid] : l_map_node_halophyid) {
        const auto size = static_cast <idx_t>(set_node_halophyid.size());
        ld[p].max_node_halophyid = std::max(ld[p].max_node_halophyid, size);
        node_halophyid_size += size + 2; // +1 for node id, +1 for count
    }
    for (const auto& [_, set_cell_halophyid] : l_map_cell_halophyid) {
        const auto size = static_cast <idx_t>(set_cell_halophyid.size());
        ld[p].max_cell_halophyid = std::max(ld[p].max_cell_halophyid, size);
        cell_halophyid_size += size + 2; // +1 for cell id, +1 for count
    }
    //*** End Compute sizes

    // -------------------------------------------------------------------------
    // Allocate output arrays
    // -------------------------------------------------------------------------
    //*** Allocate
    // Description in the header file
    auto *py_phyid_neighbor = new PyArray<idx_t, 2>(make_npy_dims(neighbor_size, 3));
    auto *py_phyid_recv = new PyArray<idx_t, 1>(make_npy_dims(recv_size));
    auto *py_phyid_send = new PyArray<idx_t, 1>(make_npy_dims(send_size));
    auto *py_node_halophyid = new PyArray<idx_t, 1>(make_npy_dims(node_halophyid_size));
    auto *py_cell_halophyid = new PyArray<idx_t, 1>(make_npy_dims(cell_halophyid_size));
    ld[p].phyid_neighbor = py_phyid_neighbor;
    ld[p].phyid_recv = py_phyid_recv;
    ld[p].phyid_send = py_phyid_send;
    ld[p].node_halophyid = py_node_halophyid;
    ld[p].cell_halophyid = py_cell_halophyid;
    //*** End Allocate

    // -------------------------------------------------------------------------
    // Fill phyid_neighbor, phyid_recv, phyid_send
    // -------------------------------------------------------------------------
    // Note: if a partition only receives from a neighbour (never sends), the
    // neighbour entry is still created so the neighbour list stays symmetric.
    //*** start phyid_neighbor, phyid_recv, phyid_send
    neighbor_size = 0;
    recv_size     = 0;
    send_size     = 0;

    for (const auto& [neighbor_part_id, set_halophyid]: l_map_phyid_recv) {
        //set_halophyid: Elements that will be received
        //set_intphyid: Elements that will be sent
        const auto &set_intphyid = ld[neighbor_part_id].map_phyid_recv.at(p);

        // phyid_neighbor row: [neighbour_partition_id, nb_phyids_we_send, nb_phyids_we_recv]
        py_phyid_neighbor->get2(neighbor_size, 0) = neighbor_part_id;
        py_phyid_neighbor->get2(neighbor_size, 1) = static_cast<idx_t>(set_intphyid.size());
        py_phyid_neighbor->get2(neighbor_size, 2) = static_cast<idx_t>(set_halophyid.size());
        neighbor_size++;

        // phyid_recv: global phyid of each exterior physical face we receive.
        for (const idx_t halophyid: set_halophyid) {
            py_phyid_recv->get(recv_size) = halophyid;
            //py_phyid_recv->get(recv_size) = ld[neighbor_part_id].map_phyid.at(halophyid);

            // Populate phyid_recv offset for node/cell_halophyid
            map_halophyid[halophyid] = recv_size;
            recv_size++;
        }

        // phyid_send: local phyid of each physical face we send to this neighbour.
        for (const idx_t intphyid: set_intphyid) {
            py_phyid_send->get(send_size) = l_map_phy_faces.at(intphyid);
            send_size++;
        }
    }
    //*** end phyid_neighbor, phyid_recv, phyid_send

    // -------------------------------------------------------------------------
    // Fill node_halophyid
    // -------------------------------------------------------------------------
    // Layout per node: [local_node_id, count, phyid_recv_idx_0, phyid_recv_idx_1, ...]
    //*** start node_halophyid
    node_halophyid_size = 0;
    for (const auto& [local_node_id, set_node_halophyid] : l_map_node_halophyid) {
        py_node_halophyid->get(node_halophyid_size) = local_node_id;
        node_halophyid_size++;
        py_node_halophyid->get(node_halophyid_size) = static_cast<idx_t>(set_node_halophyid.size());
        node_halophyid_size++;
        for (const idx_t halophyid: set_node_halophyid) {
            // Translate global phyid → phyid_recv offset.
            py_node_halophyid->get(node_halophyid_size) = map_halophyid.at(halophyid);
            node_halophyid_size++;
        }
    }
    //*** end node_halophyid

    // -------------------------------------------------------------------------
    // Fill cell_halophyid
    // -------------------------------------------------------------------------
    // Layout per cell: [local_cell_id, count, phyid_recv_idx_0, phyid_recv_idx_1, ...]
    //*** start cell_halophyid
    cell_halophyid_size = 0;
    for (const auto& [local_cell_id, set_cell_halophyid] : l_map_cell_halophyid) {
        py_cell_halophyid->get(cell_halophyid_size++) = local_cell_id;
        py_cell_halophyid->get(cell_halophyid_size++) = static_cast<idx_t>(set_cell_halophyid.size());
        for (const idx_t halophyid: set_cell_halophyid) {
            py_cell_halophyid->get(cell_halophyid_size++) = map_halophyid.at(halophyid);
        }
    }
    //*** end cell_halophyid

    // #########################################################
    // Return l_node_oldname, l_node_halophyid
    // #########################################################

    ld[p].node_oldname = new PyArray<idx_t, 1>(make_npy_dims(l_nb_nodes));
    auto l_node_oldname = ld[p].node_oldname;
    // For each local node, store the original boundary name from the global mesh.
    for (idx_t l_id = 0; l_id < l_nb_nodes; l_id++) {
        const idx_t g_id = l_node_loctoglob->get(l_id);

        //*** node_oldname
        l_node_oldname->get(l_id) = vec_node_oldname[g_id];
    }


    // #########################################################
    // Return l_phy_faces, l_phy_faces_name
    // #########################################################

    ld[p].phy_faces = new PyArray<idx_t, 2>(make_npy_dims(l_map_phy_faces.size(), max_phy_face_nodeid + 1));
    ld[p].phy_faces_name = new PyArray<idx_t, 1>(make_npy_dims(l_map_phy_faces.size()));
    auto l_phy_faces = ld[p].phy_faces;
    auto l_phy_faces_name = ld[p].phy_faces_name;

    for (const auto &item : l_map_phy_faces) {
        const idx_t g_id = item.first; ///< global physical face id
        const idx_t l_id = item.second; ///< local  physical face id

        //*** phy_faces_name
        l_phy_faces_name->get(l_id) = phy_faces_name->get(g_id);

        //*** start phy_faces
        const auto sub_phy_faces = phy_faces->sub_array(g_id);
        auto sub_l_phy_faces = l_phy_faces->sub_array(l_id);
        for (idx_t j = 0; j < sub_phy_faces.last(); j++) {
            const idx_t nodeid = sub_phy_faces.get(j);
            // Translate global node ids → local node ids inside this partition.
            sub_l_phy_faces.get(j) = vec_map_nodes(nodeid, p);
        }
        sub_l_phy_faces.last() = sub_phy_faces.last(); // copy node count
        //*** end phy_faces
    }
}

// ============================================================================
// loop_through_nodes
// ============================================================================
/**
 * @brief Assigns global nodes to partitions and builds local node arrays.
 *
 * For each global node, this function determines which partitions "own" it —
 * a node is shared by all partitions whose cells touch it.
 *   * RETURN: (description at LocalDomainStruct.h)
 *   - ld[x].nodes local node coordinates
 *   - ld[x].node_loctoglob GlobalId -> LocalId
 *   * Also Return (create_sub_domains)
 *   - vec_map_nodes: (global_node, partition) → local_node_id.
 *   - node_is_boundary: Marks nodes shared by more than one partition as boundary.
 *   - max_local_nodes: Tracks the maximum number of local nodes in any single partition,
 *
 * @param ld               Array of partition descriptors.
 * @param part_vert        Global cell → partition assignment [nb_cells].
 * @param node_cellid      Global node → incident cell list [nb_nodes, max+1].
 * @param nodes            Global node coordinates [nb_nodes, 3].
 * @param node_is_boundary Output: true/1 if this global node touches > 1 partition.
 * @param vec_map_nodes    Output: (global_node, partition) → local_node_id.
 * @param max_local_nodes  Output: maximum local node count over all partitions.
 * @param nb_parts         Number of partitions.
 */
static void loop_through_nodes(
    LocalDomainStruct *ld,
    PyArray<idx_t, 1> *part_vert,
    PyArray<idx_t, 2> *node_cellid,
    PyArray<fdx_t, 2> *nodes,
    std::vector<int8_t> &node_is_boundary,
    VecMapNodes &vec_map_nodes,
    idx_t &max_local_nodes,
    const idx_t nb_parts
) {

    // #########################################################
    // Counting
    // #########################################################
    std::vector<idx_t> parts; // temporarily vector to store node neighboring parts ID
    std::vector<int8_t> part_seen(nb_parts, false); // part_seen[p] == true means partition p was already seen for the current node.
    std::vector<idx_t> local_nodes_counter(nb_parts, 0); // determine the number of nodes for each parts


    for (idx_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        for (idx_t j = 0; j < sub_node_cellid.last(); j++) {
            const idx_t neighbor_cell = sub_node_cellid.get(j);
            const idx_t neighbor_part = part_vert->get(neighbor_cell);

            if (part_seen[neighbor_part] == false) {
                // record this partition the first time we see it
                parts.push_back(neighbor_part);
            }

            part_seen[neighbor_part] = true;
        }
        for (idx_t j = 0; j < parts.size(); j++) {
            const idx_t part = parts[j];
            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;

            // Reset part_seen to 0 for reuse in next node iteration.
            parts[j] = 0;
            part_seen[part] = false;
        }

        // reset the size of the vector
        parts.clear();
    }

    // #########################################################
    // Allocating
    // #########################################################
    for (idx_t i = 0; i < nb_parts; i++) {
        const idx_t nb_nodes = local_nodes_counter[i];

        ld[i].nodes = new PyArray<fdx_t, 2>(make_npy_dims(nb_nodes, 3), true);
        ld[i].node_loctoglob = new PyArray<idx_t, 1>(make_npy_dims(nb_nodes));
        max_local_nodes = std::max(max_local_nodes, nb_nodes); // track maximum for create_sub_domains::vec_max size
    }

    // #########################################################
    // Filling
    // #########################################################
    std::fill(part_seen.begin(), part_seen.end(), false);
    std::fill(local_nodes_counter.begin(), local_nodes_counter.end(), 0);

    for (idx_t i = 0; i < node_cellid->shape[0]; i++) {
        auto sub_node_cellid = node_cellid->sub_array(i);
        const auto sub_nodes = nodes->sub_array(i);

        for (idx_t j = 0; j < sub_node_cellid.last(); j++) {
            const idx_t neighbor_cell = sub_node_cellid.get(j);
            const idx_t neighbor_part = part_vert->get(neighbor_cell);
            const idx_t local_nodeid = local_nodes_counter[neighbor_part];

            if (part_seen[neighbor_part] == false) {
                // record this partition the first time we see it
                parts.push_back(neighbor_part);

                //*** assign node_loctoglob and vec_map_nodes
                ld[neighbor_part].node_loctoglob->get(local_nodeid) = i;
                vec_map_nodes.insert(i, neighbor_part, local_nodeid);
            }

            part_seen[neighbor_part] = true;


            //*** assign nodes
            const auto sub_l_nodes = ld[neighbor_part].nodes->sub_array(local_nodeid);
            for (idx_t k = 0; k < nodes->shape[1]; k++) {
                sub_l_nodes.get(k) = sub_nodes.get(k);
            }
        }

        for (idx_t j = 0; j < parts.size(); j++) {
            const idx_t part = parts[j];
            if (parts.size() > 1) {
                // Node is shared by multiple partitions → it lies on a boundary. (Not physical boundaries)

                //*** assign nodes_is_boundary
                node_is_boundary[i] = true;
            }

            // count the number of nodes for every sub_domain
            local_nodes_counter[part]++;

            // reset part_seen
            parts[j] = 0;
            part_seen[part] = false;
        }

        // reset the size of the vector
        parts.clear();
    }
}

// ============================================================================
// loop_through_physical_faces
// ============================================================================
/**
 * @brief Assigns physical boundary faces to partitions.
 *
 * For each physical face in the global mesh, this function:
 *   - Finds the unique interior cell attached to it via intersect_arr utils.
 *   - Derives the owning partition from that cell's assignment.
 *   * RETURN: (description at LocalDomainStruct.h)
 *    * ld[x].map_phyid global→local face id
 *    * ld[x].max_phy_face_nodeid
 *   * Also Return (create_sub_domains)
 *    * vec_node_oldname
 *    * part_phyid
 *
 * Throws std::runtime_error if any physical face is not attached to a domain cell
 * (which indicates a malformed input mesh).
 *
 * @param ld              Array of partition descriptors.
 * @param part_vert       Global cell → partition id [nb_cells].
 * @param node_cellid     Global node → incident cells [nb_nodes, max+1].
 * @param phy_faces       Global Physical face connectivity [nb_phy_faces, max_nodeid+1].
 * @param phy_faces_name  Global Physical face name/tag [nb_phy_faces].
 * @param vec_node_oldname Output: per global node, its original boundary name (minimum name if two physical faces hit the same node have different names).
 * @param part_phyid      Output: Global physical face -> PartitionId.
 */
static void loop_through_physical_faces(
    LocalDomainStruct *ld,
    PyArray<idx_t, 1> *part_vert,
    PyArray<idx_t, 2> *node_cellid,
    PyArray<idx_t, 2> *phy_faces,
    PyArray<idx_t, 1> *phy_faces_name,
    std::vector<idx_t> &vec_node_oldname,
    std::vector<idx_t> &part_phyid
) {

    std::vector<idx_t> intersect_cell(2); // holds at most 2 cells touching a face
    idx_t total_nb_phyfaces = 0;

    for (idx_t i = 0; i < phy_faces->shape[0]; i++) {
        auto phy_face = phy_faces->sub_array(i);
        const idx_t name = phy_faces_name->get(i);
        const idx_t size = phy_face.last(); // number of nodes in this face

        // Find the cell(s) sharing all nodes of this physical face.
        // A domain face touches at most 2 cells; a physical (boundary) face touches exactly 1.
        intersect_arr(node_cellid, &phy_face, size, intersect_cell);

        if (intersect_cell[0] != -1) {
            const idx_t cell_id = intersect_cell[0];
            const idx_t p = part_vert->get(cell_id); // partition that owns this face

            //*** local_max_phy_face_nodeid
            ld[p].max_phy_face_nodeid = std::max(size, ld[p].max_phy_face_nodeid);

            //*** local_max_phy_face_nodeid
            // Insert into this partition's global→local face map.
            // The local id is simply the current size before insertion.
            auto &tmp_map = ld[p].map_phyid;
            tmp_map[i] = static_cast<idx_t>(tmp_map.size());

            //*** part_phyid
            part_phyid[total_nb_phyfaces] = p;
            total_nb_phyfaces++;
        }

        // Track the minimum boundary name for each node on this face.
        for (idx_t j = 0; j < size; j++) {
            const idx_t nodeid = phy_face.get(j);
            //*** vec_node_oldname
            if (vec_node_oldname[nodeid] == 0 or vec_node_oldname[nodeid] > name)
                vec_node_oldname[nodeid] = name;
        }
    }
    if (total_nb_phyfaces != phy_faces->shape[0]) {
        throw std::runtime_error("Bad input mesh, One of the physical faces is not attached to any domain cell.");
    }
}

// ============================================================================
// loop_through_cells
// ============================================================================
/**
 * @brief Assigns cells to partitions and discovers inter-partition relationships.
 *
 * For every global cell this function:
 *   1. Determines its owning partition from part_vert.
 *   2. Writes local cells, cells_type, and cell_loctoglob arrays.
 *   3. For every node of the cell, if the node is on a boundary:
 *      - Finds all neighbouring cells belonging to *other* partitions.
 *      - Adds entries to map_int_halos (interior halos: local cells this partition
 *        must send to neighbours).
 *      - Adds entries to vec_node_halos (local_node → touching exterior halo cell).
 *      - Maintains max_halo_cell_nodeid.
 *   4. For physical boundary faces:
 *      - If the face belongs to a different partition → fills map_phyid_recv,
 *        map_node_halophyid, and map_cell_halophyid.
 *      - Otherwise → increments local physical face counters.
 *   5. Tracks max_cell_nodeid, max_cell_faceid, max_face_nodeid, max_cell_halonid,
 *      max_node_phyid, max_cell_phyid.
 *
 *      RETURN: (description at LocalDomainStruct.h)
 *          ld[x].cells
 *          ld[x].cells_type
 *          ld[x].cell_loctoglob
 *          ld[x].max_cell_nodeid
 *          ld[x].max_cell_faceid
 *          ld[x].max_face_nodeid
 *          ld[x].max_cell_halonid
 *          ld[x].max_halo_cell_nodeid
 *          ld[x].map_int_halos
 *          ld[x].map_phyid_recv
 *          ld[x].map_node_halophyid
 *          ld[x].map_cell_halophyid
 *          ld[x].vec_node_halos
 *          ld[x].max_node_phyid
 *          ld[x].max_cell_phyid
 * @param ld              Array of partition descriptors (read/write).
 * @param part_vert       Global cell → partition id [nb_cells].
 * @param node_cellid     Global node → incident cells [nb_nodes, max+1].
 * @param cells           Global cell-node connectivity [nb_cells, max_nodeid+1].
 * @param cells_type      Global cell geometric type [nb_cells] see (enum CELL_TYPE).
 * @param node_phyid      Global node → adjacent physical face ids [nb_nodes, max+1].
 * @param node_is_boundary Whether each global node touches > 1 partition.
 * @param vec_map_nodes   (global_node, partition) → local_node_id.
 * @param part_phyid      Global physical face -> PartitionId.
 * @param nb_parts        Number of partitions.
 */
static void loop_through_cells(
LocalDomainStruct *ld,
PyArray<idx_t, 1> *part_vert,
PyArray<idx_t, 2> *node_cellid,
PyArray<idx_t, 2> *cells,
PyArray<int8_t, 1> *cells_type,
PyArray<idx_t, 2> *node_phyid,
const std::vector<int8_t> &node_is_boundary,
const VecMapNodes &vec_map_nodes,
const std::vector<idx_t> &part_phyid,
const idx_t nb_parts
) {

    std::vector<idx_t> local_nb_cells(nb_parts, 0); // for every local domain count the number of local cells
    std::vector<idx_t> visited_phyid(part_phyid.size(), -1); // used to count max_cell_phyid

    // #########################################################
    // Counting — compute per-partition cell count and cell-type dimensional limits.
    // #########################################################
    for (idx_t i = 0; i < cells->shape[0]; i++) {
        const idx_t part = part_vert->get(i);
        const int8_t cell_type = cells_type->get(i);
        const auto max_info = get_max_info(cell_type); // [max_faces, max_nodes_per_face, max_nodes]

        //*** max_cell_faceid, max_face_nodeid, max_cell_nodeid
        ld[part].max_cell_faceid = std::max(max_info[0], ld[part].max_cell_faceid);
        ld[part].max_face_nodeid = std::max(max_info[1], ld[part].max_face_nodeid);
        ld[part].max_cell_nodeid = std::max(max_info[2], ld[part].max_cell_nodeid);
        local_nb_cells[part]++;
    }

    // #########################################################
    // Allocating
    // #########################################################
    for (idx_t i = 0; i < nb_parts; i++) {
        const idx_t nb_cells = local_nb_cells[i];
        const idx_t max_cell_nodeid = ld[i].max_cell_nodeid;

        ld[i].cells = new PyArray<idx_t, 2>(make_npy_dims(nb_cells, max_cell_nodeid + 1));
        ld[i].cells_type = new PyArray<int8_t, 1>(make_npy_dims(nb_cells));
        ld[i].cell_loctoglob = new PyArray<idx_t, 1>(make_npy_dims(nb_cells));

    }

    // #########################################################
    // Filling — populate cells, discover halos and physical boundary relationships.
    // #########################################################

    // i_visited[neighbour_cell] == g_id  means this neighbour was already counted
    // as a halo neighbour of cell g_id in the current cell's loop (dedup guard).
    std::vector<idx_t> i_visited(cells->shape[0], -1);
    std::fill(local_nb_cells.begin(), local_nb_cells.end(), 0);

    for (idx_t g_id = 0; g_id < cells->shape[0]; g_id++) {
        const idx_t part = part_vert->get(g_id);
        const idx_t l_id = local_nb_cells[part]; // this cell's local id
        const int8_t cell_type = cells_type->get(g_id);
        auto &map_int_halos = ld[part].map_int_halos;

        //*** assign local_cells_type
        ld[part].cells_type->get(l_id) = cell_type;

        //*** assign l_cell_loctoglob
        ld[part].cell_loctoglob->get(l_id) = g_id;


        const auto sub_cells = cells->sub_array(g_id);
        auto sub_l_cells = ld[part].cells->sub_array(l_id);
        idx_t nb_cell_halonid = 0; // number of distinct halo-neighbour cells
        idx_t nb_cell_phyid = 0; // number of distinct local physical faces

        // Iterate over the nodes of this cell.
        for (idx_t j = 0; j < sub_cells.last(); j++) {
            const idx_t nodeid = sub_cells.get(j);
            const idx_t local_nodeid = vec_map_nodes(nodeid, part); // get local node id

            //*** assign cells
            sub_l_cells.get(j) = local_nodeid; // store local node id in local cell array

            // Only boundary nodes can introduce halo relationships.
            if (node_is_boundary[nodeid]) {
                auto sub_node_cellid = node_cellid->sub_array(nodeid);
                for (idx_t k = 0; k < sub_node_cellid.last(); k++) {
                    const idx_t neighbor_cell = sub_node_cellid.get(k);
                    const idx_t neighbor_part = part_vert->get(neighbor_cell);
                    const idx_t nb_neighbor_cell_nodeid = cells->last2(neighbor_cell);

                    // --- map_int_halos ---
                    // If the neighbour cell belongs to a different partition and has not
                    // been counted yet for this cell g_id, register l_id as an interior
                    // halo that must be sent to neighbor_part.
                    if (neighbor_part != part and i_visited[neighbor_cell] != g_id) {
                        i_visited[neighbor_cell] = g_id; // mark as counted for this cell
                        nb_cell_halonid++;

                        //*** start map_int_halos
                        // map_int_halos[neighbor_part] stores local cell IDs to send.
                        // Dedup: only append if l_id is not the last entry already to prevent multiple insertion.
                        auto &vec_int_halos = map_int_halos[neighbor_part];
                        if (vec_int_halos.empty() or vec_int_halos.back() != l_id) {
                            vec_int_halos.push_back(l_id);
                        }
                        //*** end map_int_halos

                    }

                    // --- vec_node_halos ---
                    // For every node of this cell that lies on the boundary with another
                    // partition, record (local_node_id_in_neighbour, g_id_of_this_cell)
                    // in the neighbour's vec_node_halos.  This lets the neighbour know
                    // which of its boundary nodes touch which exterior halo cell.
                    if (neighbor_part != part) {
                        //*** vec_node_halos
                        const idx_t l_nodeid = vec_map_nodes(nodeid, neighbor_part);
                        auto &vec_node_halos = ld[neighbor_part].vec_node_halos;

                        // Dedup: skip if last pair is already (l_nodeid, g_id).
                        if (const size_t size = vec_node_halos.size(); size == 0 or vec_node_halos[size - 1] != g_id or vec_node_halos[size - 2] != l_nodeid) { // to prevent duplication
                            vec_node_halos.push_back(l_nodeid); // local node id in neighbour
                            vec_node_halos.push_back(g_id); // global cell id (= the exterior halo)
                        }
                        //*** end vec_node_halos

                        //*** max_halo_cell_nodeid
                        // Track the widest halo cell (for array sizing in create_halos).
                        ld[part].max_halo_cell_nodeid = std::max(ld[part].max_halo_cell_nodeid, nb_neighbor_cell_nodeid);
                    }
                }
            }

            //*** start map_phyid_recv, map_node_halophyid
            // --- Physical boundary relationships ---
            // node_phyid lists the global ids of all physical faces adjacent to this node.
            // Note: placed here (inside loop_through_cells, not loop_through_nodes) because
            //       part_phyid is only fully populated after loop_through_physical_faces runs and a node can belong to more than one subdomains.
            auto sub_node_phyid = node_phyid->sub_array(nodeid);
            idx_t nb_node_phyid = 0;

            for (idx_t k = 0; k < sub_node_phyid.last(); k++) {
                const idx_t phy_id = sub_node_phyid.get(k);
                const idx_t phy_id_part = part_phyid[phy_id]; // partition owning this face

                if (part != phy_id_part) {
                    // It is ok to use map and set here the access is rare, does not affect performance.
                    // Physical face belongs to a different partition → register as exterior phyid.
                    ld[part].map_phyid_recv[phy_id_part].insert(phy_id);
                    //Create a neighborship, this trait the case when this partition only receive and does not send to a neighbor.
                    ld[phy_id_part].map_phyid_recv[part]; // default-constructs if missing
                    ld[part].map_node_halophyid[local_nodeid].insert(phy_id);
                    ld[part].map_cell_halophyid[l_id].insert(phy_id);
                } else {
                    // Physical face belongs to this partition → count as local phyid.
                    nb_node_phyid++;
                    if (visited_phyid[phy_id] != g_id) {
                        visited_phyid[phy_id] = g_id;
                        nb_cell_phyid++;
                    }
                }
            }
            ld[part].max_node_phyid = std::max(ld[part].max_node_phyid, nb_node_phyid);
            //*** End map_phyid_recv, map_node_halophyid
        } // end node loop

        ld[part].max_cell_phyid = std::max(ld[part].max_cell_phyid, nb_cell_phyid);
        sub_l_cells.last() = sub_cells.last();

        //*** assign max_cell_halonid
        ld[part].max_cell_halonid = std::max(ld[part].max_cell_halonid, nb_cell_halonid);

        local_nb_cells[part]++;
    } // end cell loop
}

// ============================================================================
// get_result_as_py_list
// ============================================================================
/**
 * @brief Packs each partition's data into a Python tuple and returns a Python list.
 *
 * Calls LocalDomainStruct::create_tuple() for each partition to wrap all NumPy
 * arrays into a Python tuple, then transfers ownership to a Python list.
 * After this call, ld[i].tuple_res is set to nullptr to prevent double-free.
 *
 * @param ld       Array of fully-populated partition descriptors.
 * @param nb_parts Number of partitions.
 * @return         A new Python list of nb_parts tuples (each is one partition's data).
 */
static PyObject *get_result_as_py_list(LocalDomainStruct *ld, const idx_t nb_parts) {
    PyObject *py_list_result = PyList_New(nb_parts);
    if (!py_list_result) {
        throw std::bad_alloc();
    }
    for (idx_t i = 0; i < nb_parts; i++) {
        ld[i].create_tuple(); // wraps all PyArray pointers into a Python tuple
        PyList_SET_ITEM(py_list_result, i, ld[i].tuple_res);

        // The ownership transferred to list.
        ld[i].tuple_res = nullptr;
    }

    return py_list_result;
}

// ============================================================================
// create_sub_domains   (public entry point)
// ============================================================================
/**
 * @brief Top-level entry point: splits the global mesh into @p nb_parts local subdomains.
 *
 * Execution pipeline:
 *   Phase 1 (topology analysis):
 *     1. loop_through_nodes          — node distribution + local node arrays.
 *     2. loop_through_physical_faces — physical face ownership.
 *     3. loop_through_cells          — cell distribution + halo / phyid discovery.
 *   Phase 2 (communication structures, per partition):
 *     4. create_halos                — halo exchange arrays.
 *     5. create_phy                  — physical boundary communication arrays.
 *
 * Scratch/temporary data (freed when this function returns):
 *   - vec_node_oldname  — original boundary name per global node.
 *   - part_phyid        — Global physical face -> PartitionId.
 *   - node_is_boundary  — flag: global node touches > 1 partition.
 *   - vec_map_nodes     — VecMapNodes: (global_node, partition) → local_node_id.
 *   - vec_cell_to_halo  — global_cell_id → exterior halo index (reused across partitions).
 *   - vec_max           — scratch counter for max_node_haloid (size = max_local_nodes).
 *
 * @param ld           Pre-allocated array of nb_parts default-constructed LocalDomainStruct.
 * @param part_vert    Cell → partition assignment [nb_cells] (output of METIS/parMETIS).
 * @param node_cellid  Node → incident cell list [nb_nodes, max+1].
 * @param nodes        Node coordinates [nb_nodes, 3].
 * @param cells        Cell-node connectivity [nb_cells, max_nodeid+1].
 * @param cells_type   Cell geometric types [nb_cells].
 * @param phy_faces    Physical face connectivity [nb_phy_faces, max_nodeid+1].
 * @param phy_faces_name Physical face name tags [nb_phy_faces].
 * @param node_phyid   Node → adjacent physical faces [nb_nodes, max+1].
 * @param nb_parts     Number of partitions.
 * @param dim          Spatial dimension (2 or 3).
 * @return             Python list of nb_parts tuples, each holding one partition's arrays.
 */
PyObject *create_sub_domains(
    LocalDomainStruct *ld,
    PyArray<idx_t, 1> *part_vert,
    PyArray<idx_t, 2> *node_cellid,
    PyArray<fdx_t, 2> *nodes,
    PyArray<idx_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    PyArray<idx_t, 2> *phy_faces,
    PyArray<idx_t, 1> *phy_faces_name,
    PyArray<idx_t, 2> *node_phyid,
    const idx_t nb_parts,
    const idx_t dim
    ) {
    DEBUG_PRINT_INSTANT("nb_parts = %d\n", static_cast<int32_t>(nb_parts));
    // max_local_nodes is computed by loop_through_nodes and used to size vec_max.
    idx_t max_local_nodes = 0;

    std::vector<idx_t> vec_node_oldname(nodes->shape[0]); ///< per global node: original boundary name
    std::vector<idx_t> part_phyid(phy_faces->shape[0]); // store the physical face partition ID
    std::vector<int8_t> node_is_boundary(nodes->shape[0], false);
    VecMapNodes vec_map_nodes(nodes->shape[0]); ///< flat cache-friendly node-partition map
    std::vector<idx_t> vec_cell_to_halo(cells->shape[0], -1); ///< global_cell_id → exterior halo index (reused across partitions).
    
    //part1
    DEBUG_TIME_IT("");
    loop_through_nodes(ld, part_vert, node_cellid, nodes, node_is_boundary, vec_map_nodes, max_local_nodes, nb_parts);
    loop_through_physical_faces(ld, part_vert, node_cellid, phy_faces, phy_faces_name, vec_node_oldname, part_phyid);
    loop_through_cells(ld, part_vert, node_cellid, cells, cells_type, node_phyid, node_is_boundary, vec_map_nodes, part_phyid, nb_parts);
    DEBUG_TIME_IT("loop_through");
    //part2
    DEBUG_TIME_IT("");
    std::vector<idx_t> vec_max(max_local_nodes, 0);
    for (idx_t p = 0; p < nb_parts; p++) {
        create_halos(ld, cells, nodes, vec_cell_to_halo, vec_max, p, dim);
        create_phy(ld, p, phy_faces_name, phy_faces, vec_node_oldname, vec_map_nodes);
    }
    DEBUG_TIME_IT("create_halos, create_phy");

    return get_result_as_py_list(ld, nb_parts);
}
