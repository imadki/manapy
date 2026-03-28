import manapy_part32_32
import manapy_part32_64
import manapy_part64_32
import manapy_part64_64
from manapy.backends.types import FLOAT_TYPE, INT_TYPE

api_dic = {
    "int32": {
        "float32": manapy_part32_32,
        "float64": manapy_part32_64,
    },
    "int64": {
        "float32": manapy_part32_32,
        "float64": manapy_part32_64,
    }
}

api = api_dic[INT_TYPE][FLOAT_TYPE]


# Docs and more information are in src/py_manapy_part.cpp, includes/LocalDomainStruct.h


# 1. METIS_PartGraphKway ------------------------------------------- #
#nb_parts: int32
def make_n_part_graph_k_way(graph, nb_part):
    """Partition *graph* into *nb_part* parts with METIS_PartGraphKway."""
    return api.make_n_part_graph_k_way(graph, nb_part)


# 2. METIS_PartMeshDual -------------------------------------------- #
#nb_parts: int32, n_common: int32
def make_n_part_mesh_dual(cells, nb_parts, n_common):
    """Dual-mesh partitioning via METIS_PartMeshDual."""
    return api.make_n_part_mesh_dual(cells, nb_parts, n_common)


# 3. METIS_PartMeshNodal ------------------------------------------- #
#nb_parts: int32
def make_n_part_mesh_nodal(cells, nb_parts):
    """Nodal-mesh partitioning via METIS_PartMeshNodal."""
    return api.make_n_part_mesh_nodal(cells, nb_parts)


# 4. Domain builder ------------------------------------------------ #
def create_local_domains(
        part_vert,
        node_cellid,
        node_phyid,
        cells,
        cells_type,
        nodes,
        phy_faces,
        phy_faces_name,
        nb_parts, #int32
        dim, #int32
):
    """Split the mesh into *nb_parts* local domains and return their data.

    Returns
    -------
    list[tuple]
        A Python list of length `nb_parts`.
        The `p`-th element (`parts[p]`) is a 22-item tuple containing
        every array that belongs to partition `p`.

        0. nodes               - fdx_t (n_nodes_p, ndim)
        1. cells               - idx_t (n_cells_p, max_nodes_per_cell)
        2. cells_type          - int8  (n_cells_p,)
        3. phy_faces           - idx_t (n_phy_faces_p, max_nodes_per_face)
        4. phy_faces_name      - idx_t (n_phy_faces_p,)
        5. cell_loctoglob      - idx_t (n_cells_p,)
        6. node_loctoglob      - idx_t (n_nodes_p,)
        7. node_oldname        - idx_t (n_nodes_p,)
        8. halo_neighsub       - idx_t (2, n_neigh_parts_p)
        9. node_halos          - idx_t (2 * n_ext_halo_nodes_p,)
       10. halo_halosext       - idx_t (n_halos_p, max_cell_nodeid + 2)
       11. halo_halosint       - idx_t (n_halos_int_p,)
       12. halo_centvol        - fdx_t (n_halos_p, ndim + 1)
       13. phyid_neighbor      - idx_t [[Neighbor partition ID, nb_recv, nb_send] ...]
       14. phyid_recv          - idx_t [PhyFaceGlobalId, ...]
       15. phyid_send          - idx_t [PhyFaceLocalId], ...
       16. node_halophyid      - idx_t [NodeLocalId1, IndexPointToPhyId_recv, ... Size1, NodeLocalId2, ... Size2, ...., SizeN]
       17. cell_halophyid      - idx_t [...]
       18. max_node_phyid      - idx_t
       19. max_node_halophyid  - idx_t
       20. max_cell_phyid      - idx_t
       21. max_cell_halophyid  - idx_t
    """
    return api.create_local_domains(
        part_vert,
        node_cellid,
        node_phyid,
        cells,
        cells_type,
        nodes,
        phy_faces,
        phy_faces_name,
        nb_parts,
        dim
    )


# 5. Geometric helpers (2-D) --------------------------------------- #
def compute_cell_center_area_2d(cells, nodes, cell_area, cell_center):
    """Populate *cell_area* and *cell_center* with 2-D cell areas & centroids."""
    api.compute_cell_center_area_2d(cells, nodes, cell_area, cell_center)


# 6. Geometric helpers (3-D) --------------------------------------- #
def compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center):
    """Populate *cell_volume* and *cell_center* with 3-D cell volumes & centroids."""
    api.compute_cell_center_volume_3d(cells, nodes, cell_volume, cell_center)
