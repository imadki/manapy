# Converting `to_convert.py` to C++ (CPU only)

Source: [to_convert.py](to_convert.py) — 39 Python/Numba functions used by
`manapy.domain` to build mesh connectivity and geometry. This doc splits that
work into batches and adapts [../core/README.md](../core/README.md)'s
add-a-kernel recipe to a CPU-only target (no `common/<kernel>_common.hpp`
element function, no `gpu/*.cu`, no `*_cuda` binding).

Target layout (already scaffolded): `src/domain/domain_compute.hpp`,
`src/domain/cpu/`, `src/domain/bindings/`, plus a new `src/domain/common/`
for shared helpers.

## Per-kernel recipe

For every function that becomes a Python-callable kernel:

1. Declare it in `src/domain/domain_compute.hpp` (`ArrayView` in/out args,
   same order as the Python signature).
2. Implement it in `src/domain/cpu/<kernel>_cpu.cpp`: `#include
   "domain_compute.hpp"`, loop over elements inline. No per-kernel common
   header — this file has no GPU counterpart to share code with, so the math
   lives directly in the `.cpp`. `#include "common/domain_helpers.hpp"` if it
   needs a shared helper (Batch 0).
3. Add `src/domain/bindings/<kernel>.cpp`: `<kernel>_py(...)` taking
   `CFVec`/`CFMat`/`CIVec`/`CIMat`/... args, calling `<kernel>(...)` via
   `make_view`; `void register_<kernel>(nb::module_ &m)` with `m.def(...)`.
   A Python `raise RuntimeError(...)` becomes `throw
   std::runtime_error(...)` — nanobind turns that into a Python
   `RuntimeError` automatically, no manual translation needed.
4. Declare `register_<kernel>` in `src/domain/bindings/registry.hpp`.
5. Call `register_<kernel>(m);` from `src/domain/bindings/module.cpp` — its
   own `_core` module, separate from `src/core`'s (see "Module: manapy.domain
   is its own extension" below).
6. Add the two new files (`bindings/<kernel>.cpp`, `cpu/<kernel>_cpu.cpp`) to
   every `nanobind_add_module(_domaincore_<F>_<I> ...)` source list in
   `CMakeLists.txt` (all four precision targets).
7. Add the symbol to all four `python/manapy_domain_*/__init__.py`
   (`from ._core import (...)` block + `__all__`).

Functions only ever called by other functions in this file (never a
standalone entry point) skip steps 3–7: they become plain inline helpers in
`src/domain/common/domain_helpers.hpp` instead, with no Python binding.

Names drop the leading underscore (e.g. `_create_node_cellid` →
`create_node_cellid`) to match the `core/` convention of unprefixed public
kernel names.

## Prerequisite: `int8` array support ✅ done

`cell_type` and `b_visited` are typed `int8[:]` in Python (`ghost_i_visited`
in Batch 4's `create_ghost_tables` is a regular `int[:]`, despite the name —
double-checked against to_convert.py). `manapy_compute_types.hpp` now has
both `CI8Vec` (const, added Batch 2 for `cell_type`) and `I8Vec` (mutable,
added Batch 4 for `b_visited`). Batch 8's `create_halo_cells` also takes a
`b_visited: int8[:]` and can reuse `I8Vec` directly.

## Batches

One batch = one self-contained chunk of work, grouped by `to_convert.py`'s
own comment sections, ordered so a batch never depends on a later one.

### Batch 0 — shared internal helpers (no Python bindings) ✅ done
`src/domain/common/domain_helpers.hpp`. Written first since most later
batches include it. Verified with a standalone smoke test exercising each
function (not checked in — Batch 0 has no kernel to hang a binding/test on
yet; re-verify behavior once Batch 1+ kernels exercise these for real).

- [x] `is_in_array` (`_is_in_array`)
- [x] `binary_search` (`_binary_search`)
- [x] `intersect_face_nodes` (`_intersect_nodes`) — renamed to avoid a name
      clash with `intersect_common` below
- [x] `create_cell_faces` (`_create_cell_faces`)
- [x] `triangle_area_3d` (`_triangle_area_3d`)
- [x] `triangle_normal_3d` (`_triangle_normal_3d`) — writes into an
      out-param `normal` instead of returning a new array (no heap
      allocation at this layer, matches the rest of the codebase)
- [x] `get_phyid` (`_get_phyid`) — `phy_faces` is non-const: it sorts a row
      in place on first lookup, same side effect the Python view had
- [x] `search_halo_cell` (`_search_halo_cell`)
- [x] `intersect_common` (`_intersect`) — generic multi-array intersect

Also added: a `row(matrix, i)` helper (builds a 1D `ArrayView` over row `i`
of a 2D one, honoring stride) — needed because several Python helpers index
`some_2d_array[row]` and pass the result to a function typed for `int[:]`.

### Batch 1 — node/cell connectivity ✅ done
Each kernel got `domain_compute.hpp` declaration + `cpu/<kernel>_cpu.cpp` +
`bindings/<kernel>.cpp` + registry/module/CMake/`__init__.py` wiring, per
the recipe above. Verified end-to-end (not just compiled) against a
two-triangle mesh loaded straight from the built `.so` — node degrees,
sorted `node_cellid` rows, physical-face touch counts, and cell-cell
adjacency all matched hand-computed values.

- [x] `count_max_node_cellid`
- [x] `create_node_cellid` — reuses Batch 0's `row()` + `insertion_sort()`
- [x] `get_cell_nb_phyid`
- [x] `count_max_cell_cellnid`
- [x] `create_cell_cellnid`

Needed one addition outside `src/domain/`: `src/base/manapy_compute_types.hpp`
had no *mutable* int ndarray alias (every `src/core` kernel's int arrays are
read-only; only floats are written in place there). Added `IVec`/`IMat`
alongside the existing `CIVec`/`CIMat`.

### Batch 2 — face/cell topology ✅ done
Verified end-to-end on the same two-triangle mesh as Batch 1: 5 unique
faces (3+3 minus the shared edge), correct cell→face and
cell→neighbor-cell tables, and the shared edge (nodes 1,2) correctly
resolved to one face id with both cells listed as its neighbors.

- [x] `create_info` (uses Batch 0: `create_cell_faces`, `intersect_face_nodes`)

`cell_type` turned out to be `int8[:]` in the Python original (not caught
when this doc was first written — the int8 prerequisite below actually
starts here, not at Batch 4). Added `CI8Vec` to `manapy_compute_types.hpp`
for it. Also added two more `ArrayView` methods in `array_view.hpp`
(alongside `row()`): `as_const()`, needed here because `tmp_cell_faces.row(j)`
is a mutable view but `intersect_face_nodes` takes a `const` one.

### Batch 3 — 3D face geometry ✅ done
Verified with real coordinates: a two-triangle 2D mesh (correct edge
lengths incl. the √2 diagonal, and every normal pointing away from its
owning cell) and a single tetrahedron in 3D (correct 0.5 triangle area,
correct outward orientation on all 4 faces, tangent == node1 - node0).

- [x] `compute_face_info_2d`
- [x] `compute_face_info_3d` (uses Batch 0: `triangle_area_3d`, `triangle_normal_3d`)

Needed one more shared-file addition: `manapy_compute_types.hpp` had no
*mutable 2D float* ndarray alias either (only `FVec`, 1D — every `src/core`
kernel's float outputs are 1D). Added `FMat` for `face_center`/`face_normal`/
etc.

### Batch 4 — ghost / boundary cell tables ✅ done
`get_bf_recv_part_info` and `create_ghost_new_index` were implemented,
verified, then removed from scope at the user's request — not needed for
this port. Everything else verified end-to-end on a single-triangle mesh
(all 3 edges as boundary faces): `create_bf_cellid` correctly resolved each
physical face to (cell 0, local face index); `create_ghost_info`'s mirrored
ghost centers were cross-checked against an independently-coded numpy
reflection formula for all 3 faces; `create_ghost_tables`' node/cell ghost
membership matched hand-computed sets. The halo/partition functions
(`count_max_bcell_halophyid`, `create_bcell_halophyid`, `get_max_b_ncellid`,
`create_b_ncellid`) were checked standalone with small hand-verified inputs.

- [x] `create_bf_cellid` (uses Batch 0: `intersect_face_nodes`)
- [x] `create_ghost_info`
- [x] `create_ghost_tables`
- [x] `count_max_bcell_halophyid`
- [x] `create_bcell_halophyid`
- [x] `get_max_b_ncellid` — needed the mutable `int8` alias (`I8Vec`)
- [x] `create_b_ncellid` — same
- [x] ~~`get_bf_recv_part_info`~~ — removed, out of scope
- [x] ~~`create_ghost_new_index`~~ — removed, out of scope

One shared-file addition, following the same pattern as Batches 2-3:
`I8Vec` (mutable 1D int8 ndarray alias, for `b_visited`) in
`manapy_compute_types.hpp`.

### Batch 5 — halo ghost tables ✅ done
Verified with a small hand-built scenario (1 cell with 2 halo-phyids, 2
nodes with 1 each, 3 candidate halo cells): the flat `cell_halophyid`/
`node_halophyid` encoding unpacked correctly into `cell_haloghostid`/
`node_haloghostid`, and `ext_ghost_info_int`'s column 0 got correctly
patched with the local halo-cell index resolved from each phy_id's global
id — plus confirmed the exception path fires when that global id isn't
among a node's halo candidates.

- [x] `create_halo_ghost_tables` (uses Batch 0: `search_halo_cell`)

### Batch 6 — parallel cell-cell connectivity ✅ done, now actually parallel (OpenMP)
Originally ported as a plain sequential loop sharing one caller-provided
scratch buffer (`tmp_cell_faces`/`tmp_size_info`) across cells, like
`create_info`. Revisited later to add real parallelism: since each cell
only ever writes `cell_cellfid(i, ...)` (its own row) and only reads
`const` arrays, cells are independent — the shared scratch buffer was the
only thing blocking `#pragma omp parallel for`. Fixed by dropping
`tmp_cell_faces`/`tmp_size_info` from the signature entirely and instead
declaring small fixed-size scratch (bounded by `create_cell_faces`' max: 6
faces for a hexahedron, 4 nodes for a quad face) fresh inside the loop body
— thread-private automatically, no shared mutable state, closer to what
the Python original's `numba.prange` did (fresh local arrays per
iteration) than the sequential-scratch version was. `CMakeLists.txt` gains
an optional `find_package(OpenMP COMPONENTS CXX)`, linked only into the
`_domaincore_<F>_<I>` targets; without it the pragma is just ignored by
the compiler and the loop runs sequentially — still correct either way.

Verified: the original two-triangle and single-triangle cases still pass
with the new signature; a larger 78-cell triangulated-strip mesh, forced to
4 OpenMP threads (`OMP_NUM_THREADS=4`), matches `create_info`'s
independently-computed `cell_cellfid` exactly; 5 repeated parallel runs on
that mesh are bit-identical (no race).

- [x] `create_cellfid`

### Batch 7 — face naming ✅ done
Verified on a hand-built single-triangle boundary (3 physical faces named
10/20/5): `define_node_oldname`'s "smallest name wins" rule produced the
correct per-node result; `define_face_name` correctly resolved all 3 faces
to their physical-face id/name, correctly overrode `face_name` (but not
`face_oldname`) to 10 for a face flagged as on a halo boundary, and
correctly left an unmatched face at `phyid = -1` / name 0.

- [x] `define_node_oldname`
- [x] `define_face_name` (uses Batch 0: `get_phyid`)

### Batch 8 — halo cells ✅ done
Verified on the two-triangle mesh from earlier batches, with nodes 1 and 3
synthetically given a shared halo neighbor (cell 0 in an adjacent
partition): `node_haloid` unpacked correctly from the flat `node_halos`
encoding, `face_haloid` correctly resolved to that halo cell only for the
one edge bordering it (edge (1,3)) and -1 everywhere else, and
`cell_halonid` correctly deduplicated the halo neighbor for cell 1 (which
reaches it through both of its nodes).

- [x] `create_halo_cells` (uses Batch 0: `intersect_common`, `is_in_array`)

### Batch 9 — face-gradient diamond geometry ✅ done
Verified against an independently-written numpy transcription of the
Python original, exercising all 4 face-type branches (interior/name=0,
periodic/name=11, halo/name=10, physical-boundary via face_to_phyid) for
both 2D and 3D on a small synthetic mesh — every output (air_diamond,
param1..4, f1..4) matched to floating-point tolerance. Also confirmed the
3D version's exception path fires for a face that resolves to neither a
known face_name nor a valid face_to_phyid.

- [x] `face_gradient_info_2d`
- [x] `face_gradient_info_3d`

### Batch 10 — FV face geometry ✅ done
Verified against an independently-written numpy transcription of the
Python original, exercising interior/periodic-x/periodic-y/halo faces plus
a face matching none of those names (correctly falls back to
`fv_weight_left = 1.0`, `has_right = false`). Also confirmed the exception
fires when the face normal is exactly orthogonal to the left-to-right
direction (`n·d == 0`).

- [x] `fv_face_geometry`

### Batch 11 — node-based least-squares variables ✅ done
Verified against an independently-written numpy transcription covering the
full accumulation chain (cell neighbors, ghosts, a periodic image, a halo
cell, a halo-ghost) on a small 2-node synthetic mesh, for both 2D and 3D —
`node_lambda_x/y[/z]`, the `node_R_x/y[/z]` accumulators and `node_number`
all matched. Also confirmed the singular-moment-matrix exception (a node
with a single neighbor collinear with it, making `I_xy = I_yy = 0`).
Reused Batch 0's `.row()` pattern throughout; introduced one small local
lambda (`accumulate`) per function to avoid repeating the ~6-line
moment-update block 5-6 times per function.

- [x] `variables_2d`
- [x] `variables_3d`

### Batch 12 — misc geometry ✅ done — last batch, port complete
Verified on the single-triangle mesh (Batch 4) and the two-triangle mesh
(Batches 1-2): `create_normal_face_of_cell`'s flip logic checked both ways
(already-outward normals pass through unchanged; deliberately-inverted
ones flip back); `dist_ortho_function_2d` matched an independently-written
numpy reference on a mesh with both an interior and boundary faces mixed
together (`create_info`'s own face_cellid used to split the gather lists).

`cell_nf` is `float[:,:,:]` (3D) in the Python original — the first
`ArrayView` rank-3 use in this port. Added `operator()(i,j,k)` to
`ArrayView` (`array_view.hpp`, same `requires(NDIM==3)` pattern as the
existing 1D/2D overloads) and `FTensor` (mutable 3D float ndarray alias) to
`manapy_compute_types.hpp`.

- [x] `create_normal_face_of_cell`
- [x] ~~`distance_2d`~~ — removed, out of scope
- [x] `dist_ortho_function_2d`

**All 12 batches done.** 27 of `to_convert.py`'s 39 functions are ported as
Python-callable kernels in `manapy_domain_<F>_<I>`; 9 are Batch 0's
internal-only helpers; 3 (`get_bf_recv_part_info`, `create_ghost_new_index`,
`distance_2d`) were dropped from scope at the user's request.

### Post-port addition — periodic boundary connectivity kernels ✅ done
Not part of `to_convert.py`'s original 39: three more Python-callable
kernels for building `cell_shift`/`node_periodicid` (consumed by
`variables_2d`/`variables_3d`, previously built in Python) in C++. Full
per-kernel recipe (domain_compute.hpp declaration, `cpu/<kernel>_cpu.cpp`,
`bindings/<kernel>.cpp`, registry/module/CMake/`__init__.py` wiring) —
these are standalone entry points, not internal-only helpers, despite the
leading-underscore Python names. Unlike most kernels here, the sort-based
matching scratch (lo/hi/klo/khi) scales with mesh boundary size and isn't
bounded by a small compile-time constant, so it's `std::vector`-backed
instead of a fixed-size stack buffer — fine since these run once per mesh
build, not in a per-element hot loop.

Verified end-to-end by loading the built `.so` directly (float64/int64)
and calling all three through their Python bindings: correct pairing plus
shift sign on both sides of a 2-face periodic strip, the -1 return on a
mismatched-count side, correct multi-bit `node_periodic_bits` encoding on
a shared node between two tagged faces, and correct two-sided
`node_periodicid`/`node_fill` accumulation (plus a correctly-unmatched
node) on a small 4-node fixture. No caller kernel wires these together
into a full periodic-boundary build yet — re-verify if/when one does.

- [x] `pair_periodic_faces` (`_pair_periodic_faces`)
- [x] `node_periodic_bits` (`_node_periodic_bits`)
- [x] `accum_periodic_dir` (`_accum_periodic_dir`)

## Module: manapy.domain is its own extension ✅ decided

Domain kernels do **not** register into `src/core`'s `_core` module. They
get their own module, package-named `manapy_domain_<float bits>_<int bits>`
(mirroring `manapy_compute_<float bits>_<int bits>`), scaffolded and
building already:

- `src/domain/bindings/registry.hpp` — `register_<kernel>` declarations
  (empty until Batch 1's first kernel).
- `src/domain/bindings/module.cpp` — `NB_MODULE(_core, m)` for this package,
  separate translation unit/target from `src/core/bindings/module.cpp`.
- `CMakeLists.txt` — a second `foreach(FBITS...) foreach(IBITS...)` block
  building `_domaincore_<F>_<I>` targets (CPU-only: no `.cu` sources, no
  `CUDA_ARCHITECTURES`/`CUDA::cudart`), installing into
  `manapy_domain_<F>_<I>/_core`.
- `python/manapy_domain_<F>_<I>/__init__.py` — one per precision pair,
  currently `__all__ = []`.
- `pyproject.toml`'s `wheel.packages` — the four new package dirs added.

Confirmed building end-to-end (`cmake -B build/vscode && cmake --build
build/vscode`, all eight targets — four `_gradcore_*`, four
`_domaincore_*`).
