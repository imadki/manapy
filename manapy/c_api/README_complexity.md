# Partitioning Module — Time & Space Complexity

## Notation

| Symbol | Meaning |
|---|---|
| **N** | Total number of global nodes |
| **C** | Total number of global cells |
| **F** | Total number of physical boundary faces |
| **P** | Number of partitions |
| **d** | Max nodes per cell (cell width; typically 4–8 for 3-D) |
| **k** | Max cells incident to a single node (node valence; bounded by mesh quality, typically 6–20) |
| **H** | Total number of exterior halo cells across all partitions (proportional to the surface area of all partition boundaries) |
| **B** | Number of boundary nodes (nodes shared by ≥ 2 partitions; H ≤ B·k) |
| **q** | Max partitions sharing a single node (≤ P, practically ≤ 4) |

In a well-partitioned mesh, **H, B ≪ N, C** since they only grow with the cut surface, not the volume.

---

## Per-Function Analysis

### `VecMapNodes` — flat node-partition map

| Operation | Time | Notes |
|---|---|---|
| Construction | O(N) | One `reserve(4)` per node |
| `insert` | O(1) amortized | `emplace_back` on a pre-reserved vector |
| `operator()` lookup | O(q) ≈ O(1) | Linear scan over ≤ q ≈ 4 entries |

**Space:** O(N · q) ≈ O(N) — one small flat vector per node, each holding at most q pairs.

> **vs. the old `std::map` version:**  
> `operator()` was O(log q) with large constant due to pointer-chasing inside the red-black tree.  
> Construction allocated O(N · q) separate heap nodes, stressing the allocator and destroying cache locality.

---

### `loop_through_nodes`

Three sub-passes over the node-cell incidence list:

| Sub-pass | Time |
|---|---|
| Counting pass | O(N · k) — for each node, iterate over incident cells |
| Allocation | O(P) — one allocation per partition |


**Total time:** O(N · k)

**Space (scratch):** O(P) for `parts`, `parts_counter`, `local_nodes_counter`.  
**Space (output):** O(N · q) for `VecMapNodes`; O(N) for `node_is_boundary`.

---

### `loop_through_physical_faces`

| Step | Time |
|---|---|
| `intersect_arr` per face | O(f · k) where f = nodes per face |
| Node name update | O(F · f) |

**Total time:** O(F · f · k)  — dominated by `intersect_arr`.

**Space (scratch):** O(1) (`intersect_cell` buffer of size 2).  
**Space (output):** O(F) for `part_phyid`; O(N) for `vec_node_oldname`.

---

### `loop_through_cells`

Two sub-passes over all global cells:

| Sub-pass | Time |
|---|---|
| Counting | O(C) |
| Allocation | O(P) |
| Filling | O(C · d · k) in the worst case |

The filling pass is the most complex. For each cell (C total):
- For each of its d nodes (O(d)):
  - If the node is a boundary node: iterate over k incident cells to find cross-partition neighbours (O(k)).
  - Lookup `vec_map_nodes` → O(q) ≈ O(1).
  - Iterate over physical face adjacency for the node (O(phyid per node) — rare, bounded).

**Total time:** O(C · d · k)  — dominated by boundary node processing.

> Interior nodes (the vast majority) skip the expensive `node_is_boundary` branch, so the effective constant is small in practice.

**Space (scratch):** O(C) for `i_visited`, `visited_phyid`, `local_nb_cells`.  
**Space (output, in `ld[p]`):** O(H) total for `map_int_halos` + `vec_node_halos` across all partitions.

---

### `create_halos` (called P times)

| Step | Time per partition p |
|---|---|
| Sparse reset of `vec_max` | O(\|vec_node_halos[p]\|) ≈ O(H_p) |
| Count exterior halos | O(nb_neighbours_of_p) |
| Fill `halo_halosext` | O(H_p · d) |
| Fill `node_halos` | O(\|vec_node_halos[p]\|) |
| Fill `halo_neighsub` / `halo_halosint` | O(nb_interior_halos_p) |
| Centroid / volume computation | O(H_p · d) |

**Total time across all P calls:** O(H · d) — linear in the total halo surface.

**Space (scratch, shared across calls):** O(N_local_max) for `vec_max` — allocated once.  
**Space (output per partition):** O(H_p · d) for `halo_halosext`; O(H_p) for remaining halo arrays.

> **Key optimization:** `vec_max` is allocated once with size `max_local_nodes` instead of O(N) and reset sparsely (only used slots), saving O(N·P) fill operations total.

---

### `create_phy` (called P times)

| Step | Time per partition p |
|---|---|
| Compute sizes | O(nb_phy_neighbours · set_size) |
| Fill `phyid_neighbor`, `phyid_recv`, `phyid_send` | O(recv_size + send_size) |
| Fill `node_halophyid` | O(node_halophyid_size) |
| Fill `cell_halophyid` | O(cell_halophyid_size) |
| Fill `node_oldname` | O(N_local_p) |
| Fill `phy_faces`, `phy_faces_name` | O(F_local_p · f) |

All steps are proportional to local sizes which sum to O(F + N) globally.

**Total time across all P calls:** O(F + N)

**Space (scratch per call):** O(recv_size) for `map_halophyid`.  
**Space (output per partition):** O(F_local_p + N_local_p).

---

## Overall Summary

### Time Complexity

| Phase | Function | Time |
|---|---|---|
| Phase 1 | `loop_through_nodes` | **O(N · k)** |
| Phase 1 | `loop_through_physical_faces` | O(F · f · k) |
| Phase 1 | `loop_through_cells` | **O(C · d · k)** |
| Phase 2 | `create_halos` × P | O(H · d) |
| Phase 2 | `create_phy` × P | O(F + N) |
| **Total** | | **O(C · d · k)** |

The dominating term is **O(C · d · k)** — linear in the mesh size when `d` and `k` are bounded constants (which they are for any quality mesh, typically d ≤ 8, k ≤ 20).

### Space Complexity

| Data | Space |
|---|---|
| `VecMapNodes` | O(N · q) ≈ **O(N)** |
| `node_is_boundary` (`int8_t[]`) | O(N) |
| `vec_node_oldname`, `part_phyid`, `vec_cell_to_halo` | O(N + C + F) |
| `vec_max` scratch (shared, size = max_local_nodes) | O(N/P) |
| Output arrays in all `ld[p]` combined | O(N + C + F + H·d) |
| **Total** | **O(N + C + F + H·d)** |

Since H ≤ C (halos are a subset of cells) and for well-partitioned meshes H ≪ C, the effective space is **O(N + C + F)** — proportional to the global mesh size with no super-linear overhead.
