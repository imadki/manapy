import manapy_part32_32
import manapy_part32_64
from manapy.backends.types import FLOAT_TYPE


api = manapy_part32_32
if FLOAT_TYPE == "float64":
  api = manapy_part32_64

# Docs and more information are in src/py_manapy_part.cpp, includes/LocalDomainStruct.h


# 1. METIS_PartGraphKway ------------------------------------------- #
def make_n_part_graph_k_way(graph, nb_part):
    """Partition *graph* into *nb_part* parts with METIS_PartGraphKway."""
    return api.make_n_part_graph_k_way(graph, nb_part)


# 2. METIS_PartMeshDual -------------------------------------------- #
def make_n_part_mesh_dual(cells, nb_parts, n_common):
    """Dual-mesh partitioning via METIS_PartMeshDual."""
    return api.make_n_part_mesh_dual(cells, nb_parts, n_common)


# 3. METIS_PartMeshNodal ------------------------------------------- #
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
        nb_parts,
        dim,
):
    """Split the mesh into *nb_parts* local domains and return their data.

    Returns
    -------
    list[tuple]
        A Python list of length `nb_parts`.
        The `p`-th element (`parts[p]`) is a 22-item tuple containing
        every array that belongs to partition `p`.

        0. nodes               - float32|float64 (n_nodes_p, ndim)
        1. cells               - int32 (n_cells_p, max_nodes_per_cell)
        2. cells_type          - int8  (n_cells_p,)
        3. phy_faces           - int32 (n_phy_faces_p, max_nodes_per_face)
        4. phy_faces_name      - int32 (n_phy_faces_p,)
        5. cell_loctoglob      - int32 (n_cells_p,)
        6. node_loctoglob      - int32 (n_nodes_p,)
        7. node_oldname        - int32 (n_nodes_p,)
        8. halo_neighsub       - int32 (2, n_neigh_parts_p)
        9. node_halos          - int32 (2 * n_ext_halo_nodes_p,)
       10. halo_halosext       - int32 (n_halos_p, max_cell_nodeid + 2)
       11. halo_halosint       - int32 (n_halos_int_p,)
       12. halo_centvol        - float32|float64 (n_halos_p, ndim + 1)
       13. phyid_neighbor      - int32 [[Neighbor partition ID, nb_recv, nb_send] ...]
       14. phyid_recv          - int32 [PhyFaceGlobalId, ...]
       15. phyid_send          - int32 [PhyFaceLocalId], ...
       16. node_halophyid      - int32 [NodeLocalId1, IndexPointToPhyId_recv, ... Size1, NodeLocalId2, ... Size2, ...., SizeN]
       17. cell_halophyid      - int32 [...]
       18. max_node_phyid      - int
       19. max_node_halophyid  - int
       20. max_cell_phyid      - int
       21. max_cell_halophyid  - int
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
