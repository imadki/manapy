# Partitioning Module — Developer Reference

This document explains the design, data structures, and execution pipeline of the
mesh partitioning module (`src/partitioning.cpp` + `includes/LocalDomainStruct.h`).

---

## Purpose

The partitioning module takes a **global mesh** (cells, nodes, physical boundary faces)
together with a **partition assignment vector** (e.g. from METIS/parMETIS) and
produces **`nb_parts` independent local subdomain descriptors**, each ready for
MPI-parallel numerical computation.

Each local subdomain is described by a `LocalDomainStruct` and includes:
- Local copies of mesh geometry (nodes, cells, physical faces).
- Global-to-local and local-to-global ID mappings.
- Halo exchange structures (cells / nodes to send and receive across partition boundaries).
- Physical boundary communication structures (for boundary conditions straddling partition cuts).

---

## File Overview

| File | Role |
|---|---|
| `src/partitioning.cpp` | Implementation: all partitioning logic |
| `includes/LocalDomainStruct.h` | Data structure: one instance per local subdomain |

---

## `LocalDomainStruct` — Field Reference

### Core Mesh Topology

| Field | Shape | Description |
|---|---|---|
| `nodes` | `[nb_nodes, 3]` | Local node coordinates `[x, y, z]` |
| `cells` | `[nb_cells, max_cell_nodeid+1]` | Local cell–node connectivity; last slot stores node count |
| `cells_type` | `[nb_cells]` | Geometric cell type (triangles, quads, tets, hexes, …) |
| `phy_faces` | `[nb_phy_faces, max_phy_face_nodeid+1]` | Local physical face connectivity (local node IDs) |
| `phy_faces_name` | `[nb_phy_faces]` | Boundary name/tag for each physical face |

### Global ↔ Local Mappings

| Field | Shape | Description |
|---|---|---|
| `cell_loctoglob` | `[nb_cells]` | Local cell ID → global cell ID |
| `node_loctoglob` | `[nb_nodes]` | Local node ID → global node ID |
| `node_oldname` | `[nb_nodes]` | Original boundary name tag for each local node |

### Halo Exchange (MPI Cell Communication)

Halos are cells from neighbouring partitions required for flux computations at shared faces.

| Field | Shape | Description |
|---|---|---|
| `halo_neighsub` | `[2, nb_neighbours]` | Row 0: neighbour partition IDs. Row 1: number of interior halos sent to each. |
| `halo_halosext` | `[nb_halos, max_halo_cell_nodeid+2]` | **Exterior halos** (received). Row: `[global_cell_id, node0, …, node_count]` |
| `halo_halosint` | `[nb_halos_int]` | **Interior halos** (local cells to send), grouped by neighbour as per `halo_neighsub` |
| `node_halos` | `[2 × nb_halo_entries]` | Flat pairs: `(local_node_id, halo_ext_index)` linking boundary nodes to exterior halos |
| `halo_centvol` | `[nb_halos, 4]` | Centroid `(x, y, z)` + volume (or area in 2D) of each exterior halo cell |
| `max_node_haloid` | scalar | Max number of exterior halo cells touching any single local node |
| `max_cell_halonid` | scalar | Max number of distinct halo-neighbour cells across any single local cell |
| `max_halo_cell_nodeid` | scalar | Max node count of any exterior halo cell |

### Physical Boundary Communication (phyid)

Used when a physical boundary face straddles a partition cut and must be
communicated between the partition that owns it and others that neighbour it.

| Field | Shape | Description |
|---|---|---|
| `phyid_neighbor` | `[nb_phy_neighbours, 3]` | `[neighbour_id, nb_send, nb_recv]` per neighbour |
| `phyid_recv` | `[nb_recv]` | Global IDs of exterior physical faces received from neighbours |
| `phyid_send` | `[nb_send]` | Local IDs of physical faces sent to neighbours |
| `node_halophyid` | flat | Per-node: `[local_node_id, count, phyid_recv_idx…]` |
| `cell_halophyid` | flat | Per-cell: `[local_cell_id, count, phyid_recv_idx…]` |

### Sizing Scalars

Used to size arrays that need a fixed column width per partition:

| Scalar | Meaning |
|---|---|
| `max_cell_nodeid` | Max nodes in any local cell |
| `max_cell_faceid` | Max faces on any local cell |
| `max_face_nodeid` | Max nodes on any face |
| `max_phy_face_nodeid` | Max nodes on any physical face |
| `max_node_phyid` | Max local phyids adjacent to any single node |
| `max_cell_phyid` | Max local phyids adjacent to any single cell |
| `max_node_halophyid` | Max exterior phyids adjacent to any single node |
| `max_cell_halophyid` | Max exterior phyids adjacent to any single cell |

### Temporary Construction Maps (freed / moved after build)

| Field | Description |
|---|---|
| `map_int_halos` | `neighbour_part → [local_cell_ids_to_send]` |
| `vec_node_halos` | Flat pairs `(local_node_id, global_halo_cell_id)` for `node_halos` |
| `map_phyid` | `global_phyid → local_phyid` |
| `map_phyid_recv` | `neighbour_part → set(exterior phyids to receive)` |
| `map_node_halophyid` | `local_node_id → set(exterior phyids)` |
| `map_cell_halophyid` | `local_cell_id → set(exterior phyids)` |

---

## Execution Pipeline (`create_sub_domains`)

```
create_sub_domains(ld, part_vert, node_cellid, nodes, cells, cells_type,
                   phy_faces, phy_faces_name, node_phyid, nb_parts, dim)
│
├── Phase 1 — topology analysis
│   ├── loop_through_nodes           → ld[p].nodes, node_loctoglob
│   │                                   VecMapNodes, node_is_boundary, max_local_nodes
│   ├── loop_through_physical_faces  → ld[p].map_phyid, max_phy_face_nodeid
│   │                                   vec_node_oldname, part_phyid
│   └── loop_through_cells           → ld[p].cells, cells_type, cell_loctoglob
│                                       map_int_halos, vec_node_halos
│                                       map_phyid_recv, map_node/cell_halophyid
│                                       various max_* scalars
│
└── Phase 2 — communication structures  (for p in 0..nb_parts-1)
    ├── create_halos(p) → halo_halosext, halo_halosint, halo_neighsub,
    │                      node_halos, halo_centvol, max_node_haloid
    └── create_phy(p)   → phyid_neighbor, phyid_recv, phyid_send,
                           node/cell_halophyid, node_oldname,
                           phy_faces, phy_faces_name
```

---

## Key Design Decisions & Optimisations

### `VecMapNodes` — flat node-partition map

Replaces `std::vector<std::map<int32_t,int32_t>>` with a vector of small flat
`std::vector<std::pair<int32_t,int32_t>>` per node.

**Why:** A node is shared by at most a handful of partitions (typically 1–4).
A linear scan over 4 pairs is faster than an O(log P) red-black tree lookup,
eliminates heap allocation per map node, and is more cache-friendly.

### `std::vector<int8_t>` instead of `std::vector<bool>`

`std::vector<bool>` uses bit-packing — every access involves a bitwise read/write.
`int8_t` stores one byte per element, making random access a simple array read.

### Shared `vec_max` scratch buffer in `create_halos`

Instead of allocating and zero-filling a `vec_max[nb_nodes]` array on every call
to `create_halos`, a single buffer sized to `max_local_nodes` is allocated once
in the caller and passed by reference.  Only the entries actually used by the
current partition are reset (sparse reset via `vec_node_halos`), avoiding an
O(nb_nodes) fill per partition.

### Three-pass pattern (Count → Allocate → Fill)

Each major loop runs in three stages to avoid dynamic resizing:
1. **Count** — determine exact array sizes.
2. **Allocate** — perform a single allocation per array.
3. **Fill** — iterate again to populate the arrays.

This eliminates reallocation overhead and produces contiguous, correctly-sized
NumPy arrays that can be handed directly back to Python.

---

## Return Value

`create_sub_domains` returns a **Python list** of `nb_parts` **Python tuples**.
Each tuple contains all the NumPy arrays for one local subdomain.
The layout of the tuple is defined by `LocalDomainStruct::create_tuple()`.
