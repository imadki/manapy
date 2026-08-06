"""METIS-backed domain partitioning compiled for float64 data and int32 indices."""

from ._core import (
    create_local_domains,
    make_n_part_graph_k_way,
    make_n_part_mesh_dual,
    make_n_part_mesh_nodal,
)

__all__ = [
    "create_local_domains",
    "make_n_part_graph_k_way",
    "make_n_part_mesh_dual",
    "make_n_part_mesh_nodal",
]
