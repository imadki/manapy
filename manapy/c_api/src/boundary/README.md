# manapy boundary conditions (`src/boundary`)

Ghost / halo-ghost boundary-condition kernels, ported from the manapy boundary
routines. Same shape as [`src/core`](../core/README.md) — one nanobind
extension — shipped as the `boundary` submodule of the same package:

- `manapy_compute_<F>_<I>.boundary` — CPU + CUDA

Each kernel fills the *ghost* values used by the gradient/limiter/solver
kernels: `w_ghost` (indexed by boundary face) for local boundary faces, and
`w_haloghost` (indexed by halo-ghost id) for the ghosts hanging off halo nodes
on a partition boundary.

## Layout

```
src/boundary/
  boundary_compute.hpp                CPU entry-point declarations
  boundary_compute.cuh                GPU launch declarations
  common/<kernel>_common.hpp          MANAPY_COMPUTE_HOST_DEVICE _face/_node routine
  common/boundary_math.hpp            portable sqrt/abs (host + device)
  cpu/<kernel>_cpu.cpp                host entry point
  gpu/<kernel>_cuda.cu                __global__ kernel + launch_<kernel>
  bindings/<kernel>.cpp               nanobind wrappers + register_<kernel>
  bindings/module.cpp                 NB_MODULE(_core)
  bindings/registry.hpp               register_* declarations
```

The include path is `src/base src/boundary`, so `"array_view.hpp"` /
`"precision.hpp"` / `"manapy_compute_types.hpp"` resolve from `src/base`, and
`"boundary_compute.hpp"` / `".cuh"` / `"common/..."` / `"bindings/registry.hpp"`
from `src/boundary`.

## Kernels

Every Python entry point has a `_cuda` twin taking CuPy device arrays.

| Python name | Applies to | Value written |
| --- | --- | --- |
| `ghost_value_dirichlet` | faces in `bc_faces` | `value[i]` |
| `ghost_value_neumann` | faces in `bc_faces` | `w_c[face_cellid[i, 0]]` |
| `ghost_value_neumannNH` | faces in `bc_faces` | `w_c[...] + cst[i] * face_dist_ortho[i]` |
| `ghost_value_nonslip` | faces in `bc_faces` | `-w_c[face_cellid[i, 0]]` |
| `haloghost_value_dirichlet` | halo ghosts of `d_halonodes` tagged `BCindex` | `value_haloghost[g]` |
| `haloghost_value_neumann` | ″ | `w_halo[ghost_ext_info_int[g, 0]]` |
| `haloghost_value_neumannNH` | ″ | `w_halo[...] + cst[g] * 2*abs(ghost_ext_info_flt[g, 0])` |
| `haloghost_value_nonslip` | ″ | `-w_halo[ghost_ext_info_int[g, 0]]` |
| `ghost_value_slip_2d` / `_3d` | faces in `bc_faces` | `U_c - 2 (U_c . n) n` |
| `haloghost_value_slip_2d` / `_3d` | halo ghosts of `d_halonodes` tagged `BCindex` | `U_halo - 2 (U_halo . n) n` |

Notes:

- The scalar conditions take *scalar* fields, so a velocity vector needs one
  call per component. Free-slip is a *vector* condition — it couples the
  components through the face normal — so it takes them all at once. The normal
  is normalised inside the kernel, so it works whether it is unit or
  area-scaled (`ghost_ext_info_flt` columns 7-8 in 2D, 7-9 in 3D for the halo
  variants).
- `bc_faces` / `d_halonodes` are *gather lists* of ids, not full-range
  counters. The halo variants additionally filter on the boundary tag
  `ghost_ext_info_int[g, 1] == BCindex`, so several boundaries sharing a node
  do not overwrite each other.
- `cst` and `face_dist_ortho` are only read by the `neumannNH` variants, and
  `face_cellid` only by the non-Dirichlet ones; they stay in every signature
  for parity with the Python API.
- `haloghost_value_dirichlet` is the odd one out: its first array is the
  prescribed **per-halo-ghost** value array (manapy's `valuehalo`, sized
  `halos.sizehaloghost` and evaluated at each ghost's face centre), hence the
  `ghost_id` indexing. The other three take the halo **cell** field and go
  through `ghost_ext_info_int[g, 0]`. The two index spaces have different sizes
  (`sizehaloghost` vs `nbhalos`), so the parameter is named `value_haloghost`
  rather than `w_halo` to keep them apart.
- A halo ghost is listed by every node of its face, so several `d_halonodes`
  entries land on the same ghost. Every halo kernel writes a value determined
  solely by `ghost_id` — `cst` included — so all contributing threads store the
  same bytes: the GPU result is deterministic and matches the CPU loop's
  regardless of visit order.

Both scalar families share one binding TU each: their four variants have
identical signatures and differ only in a compile-time `GhostValueKind` /
`HaloGhostValueKind` tag, so one templated `_face`/`_node` routine, one
templated loop and one templated kernel serve all four (same trick as
`core`'s `celltoface`). The four Python names are still distinct entry points.

## Adding a new kernel `<kernel>`

Identical to [../core/README.md](../core/README.md), with `variable_compute.hpp`
/ `.cuh` replaced by `boundary_compute.hpp` / `.cuh`:

1. Create `common/<kernel>_common.hpp` — one `MANAPY_COMPUTE_HOST_DEVICE`
   routine, e.g. `<kernel>_face(index_t i, ...)` (or `_node` for a node loop),
   shared verbatim by the CPU loop and the CUDA kernel. If the variants of one
   family differ only in the assigned expression, template it on an enum tag
   instead of writing N copies.
2. Create `cpu/<kernel>_cpu.cpp` — `#include "boundary_compute.hpp"` and the
   common header; define `void <kernel>(...)` looping over the gather list.
3. Create `gpu/<kernel>_cuda.cu` — `#include "boundary_compute.cuh"` and the
   common header; define `__global__ void <kernel>_kernel(...)` (grid-stride
   loop) and `void launch_<kernel>(..., cudaStream_t stream)`.
4. Edit `boundary_compute.hpp` — declare `void <kernel>(...)` (step 2's signature).
5. Edit `boundary_compute.cuh` — declare `launch_<kernel>(..., cudaStream_t)`.
6. Create `bindings/<kernel>.cpp` — `<kernel>_py` / `<kernel>_cuda_py` +
   `register_<kernel>`, as in core's recipe.
7. Edit `bindings/registry.hpp` — add `void register_<kernel>(nb::module_ &m);`
8. Edit `bindings/module.cpp` — add `register_<kernel>(m);`
9. Edit `CMakeLists.txt` — add the three sources to the `_boundarycore_<F>_<I>`
   target's `nanobind_add_module(...)` list.
10. Edit all four `python/manapy_compute_*/boundary/__init__.py` — add
    `<kernel>`, `<kernel>_cuda` to the `from ._core import (...)` block and to
    `__all__` (keep both sorted).
11. Rebuild:
    `cmake --build build --target _boundarycore_32_32 _boundarycore_32_64 _boundarycore_64_32 _boundarycore_64_64`
12. Verify against an independent NumPy reference, not just that it compiles.
