# manapy solvers (`src/solvers`)

PDE solver kernels, ported from the manapy solvers. Unlike `src/core` (one
module) and `src/domain` (one module), this tree holds **several** submodules,
each its own nanobind extension, shipped nested inside the same
`manapy_compute_<F>_<I>` Python package under `solvers`:

- `manapy_compute_<F>_<I>.solvers.advec` — explicit advection solver (CPU + CUDA)
- `manapy_compute_<F>_<I>.solvers.advecdiff` — explicit advection-diffusion solver (CPU + CUDA)
- `manapy_compute_<F>_<I>.solvers.diffusion` — explicit pure-diffusion solver (CPU + CUDA):
  the dissipative residual and a diffusion-only CFL time step (no convective term)
- `manapy_compute_<F>_<I>.solvers.utils` — kernels common to *all* solvers, e.g.
  initial conditions and the forward-Euler update (mixed CPU + CUDA, per-function)

(`core` and `domain` stay top-level — they aren't solvers.)

## Layout

Every stage directory is split one level further, by solver:

```
src/solvers/
  headers/<solver>/<solver>_compute.hpp     CPU entry-point declarations
  headers/<solver>/<solver>_compute.cuh     GPU launch declarations (GPU solvers only)
  common/<solver>/<kernel>_common.hpp       MANAPY_COMPUTE_HOST_DEVICE _element/_face routine
  common/helpers/                           solver-agnostic inline helpers (e.g. scatter.hpp)
  cpu/<solver>/<kernel>_cpu.cpp             host entry point
  gpu/<solver>/<kernel>_cuda.cu             __global__ kernel + launch_<kernel> (GPU solvers only)
  bindings/<solver>/<kernel>.cpp            nanobind wrappers + register_<kernel>
  bindings/<solver>/module.cpp              NB_MODULE(_core) for that submodule
  bindings/registry.hpp                     register_* declarations, shared by every submodule
```

Each submodule has its own `bindings/<solver>/module.cpp` (its own
`NB_MODULE(_core)`) and its own `_<solver>core_<F>_<I>` CMake target, but they
all share the single `bindings/registry.hpp`. The include path for every target
is `src/base src/solvers src/solvers/headers/<solver>`, so:
`"array_view.hpp"`/`"precision.hpp"`/`"manapy_compute_types.hpp"` resolve from
`src/base`, `"<solver>_compute.hpp"`/`".cuh"` from `headers/<solver>`, and
`"common/<solver>/..."`/`"common/helpers/..."`/`"bindings/registry.hpp"` from
`src/solvers`.

## Decide first

Before writing a kernel, settle three things:

1. **Which solver** it belongs to (`advec`, `advecdiff`, `diffusion`, ...), or
   whether it is common to all solvers (→ `utils`). This picks the `<solver>`
   directory and the target package.
2. **CPU-only or CPU + GPU.** This is a *per-function* choice, even within one
   submodule. CPU-only kernels follow the `src/domain` recipe (no
   `common/<kernel>_common.hpp` split, no `gpu/`, no `_cuda` binding), and their
   `headers/<solver>/<solver>_compute.hpp` declaration carries a comment saying
   the kernel is CPU-only; an unmarked declaration is assumed CPU + GPU. `utils`
   mixes both — `initialisation_gaussian_*` are CPU-only while `update_new_value`
   has a GPU counterpart — so its target is built GPU-capable.
3. **Module function or helper.** A helper only ever called by other kernels
   (never a standalone Python entry point) skips the binding/registry/module/
   Python steps — see [Helpers](#helpers).

## Adding a CPU + GPU kernel `<kernel>` (e.g. to `advec`)

Same shape as [../core/README.md](../core/README.md), with the `<solver>`
directory level added.

1. Create `common/<solver>/<kernel>_common.hpp`
   - one `MANAPY_COMPUTE_HOST_DEVICE` routine, e.g.
     `<kernel>_element(index_t i, ...)` (or `<kernel>_face` for a face loop),
     shared verbatim by the CPU loop and the CUDA kernel

2. Create `cpu/<solver>/<kernel>_cpu.cpp`
   - `#include "<solver>_compute.hpp"` and `#include "common/<solver>/<kernel>_common.hpp"`
   - define `void <kernel>(...)` looping over the elements, calling the routine

3. Create `gpu/<solver>/<kernel>_cuda.cu`
   - `#include "<solver>_compute.cuh"` and `#include "common/<solver>/<kernel>_common.hpp"`
   - define `__global__ void <kernel>_kernel(...)` (grid-stride loop) and
     `void launch_<kernel>(..., cudaStream_t stream)`

4. Edit `headers/<solver>/<solver>_compute.hpp` — declare `void <kernel>(...)`
   (same signature as step 2)

5. Edit `headers/<solver>/<solver>_compute.cuh` — declare
   `void launch_<kernel>(..., cudaStream_t stream)` (same signature as step 3)

6. Create `bindings/<solver>/<kernel>.cpp`
   - `#include "manapy_compute_types.hpp"`, `<cuda_runtime_api.h>`,
     `"<solver>_compute.cuh"`, `"<solver>_compute.hpp"`, `"bindings/registry.hpp"`
   - `<kernel>_py(...)` taking `CFVec`/`CFMat`/`FVec`/`CIVec`/`CIMat`/... args
     (see [Array types](#array-types)), calling `<kernel>(...)` via `make_view<...>`
   - `<kernel>_cuda_py(...)` taking the `D*` device aliases, calling
     `launch_<kernel>(...)`, then `cuda_check(manapy_cuda_post_launch(), "<kernel> kernel launch")`.
     That check is non-blocking: the binding returns as soon as the kernel is
     enqueued. Do NOT add a `cudaDeviceSynchronize()` — see `base/cuda_launch.hpp`
     for why, and for the `MANAPY_CUDA_SYNC=1` debug switch.
   - `void register_<kernel>(nb::module_ &m)` with `m.def("<kernel>", ...)` and
     `m.def("<kernel>_cuda", ...)`

7. Edit `bindings/registry.hpp` — add `void register_<kernel>(nb::module_ &m);`

8. Edit `bindings/<solver>/module.cpp` — add `register_<kernel>(m);` inside
   `NB_MODULE(_core, m)`

9. Edit `CMakeLists.txt` — add the three sources
   (`bindings/<solver>/<kernel>.cpp`, `cpu/<solver>/<kernel>_cpu.cpp`,
   `gpu/<solver>/<kernel>_cuda.cu`) to that solver's `nanobind_add_module(...)`
   list (advec's target is `_adveccore_<F>_<I>`)

10. Edit all four `python/manapy_compute_*/solvers/<solver>/__init__.py` — add
    `<kernel>`, `<kernel>_cuda` to the `from ._core import (...)` block and to
    `__all__` (keep both sorted)

11. Rebuild the four precision targets of that solver, e.g. for advec:
    `cmake --build build --target _adveccore_32_32 _adveccore_32_64 _adveccore_64_32 _adveccore_64_64`
    (utils' targets are `_solversutils_<F>_<I>`; each solver's block names its own)

12. [Verify, not just compile](#verify).

## Adding a CPU-only kernel `<kernel>` (e.g. to `utils`)

Follows [../domain/README.md](../domain/README.md): the whole implementation
lives in one `.cpp` — no `common/<kernel>_common.hpp` split, no `gpu/`, no
`_cuda` twin.

1. Declare it in `headers/<solver>/<solver>_compute.hpp` — `void <kernel>(...)`
   (or `real_t <kernel>(...)` if it returns a scalar). The header comment marks
   this kernel CPU-only (an unmarked declaration is assumed CPU + GPU).
2. Create `cpu/<solver>/<kernel>_cpu.cpp` — `#include "<solver>_compute.hpp"`,
   define `<kernel>(...)` looping over elements directly. Standard-library
   headers like `<cmath>` are fine here (no device build to keep portable).
3. Create `bindings/<solver>/<kernel>.cpp` — `#include "manapy_compute_types.hpp"`,
   `"bindings/registry.hpp"`, `"<solver>_compute.hpp"` (no CUDA header); define
   `<kernel>_py(...)` and `register_<kernel>(...)` with a single `m.def(...)` —
   no `_cuda` twin.
4. Edit `bindings/registry.hpp`, `bindings/<solver>/module.cpp` as in steps 7–8
   above.
5. Edit `CMakeLists.txt` — add `bindings/<solver>/<kernel>.cpp` and
   `cpu/<solver>/<kernel>_cpu.cpp` to the `_<solver>...` target. (Adding a
   CPU-only kernel to an otherwise GPU-capable target, like `utils`, needs no
   target changes beyond the sources.)
6. Edit all four `python/manapy_compute_*/solvers/<solver>/__init__.py`.
7. Rebuild + [verify](#verify).

## Helpers

A function only ever called by other kernels — never a Python entry point —
skips the binding/registry/module/Python steps. Where it goes depends on scope:

- **Solver-specific** → `common/<solver>/`. If it runs on the GPU it must be
  `MANAPY_COMPUTE_HOST_DEVICE` (e.g. `common/advec/numerical_flux_common.hpp`).
- **Common to all solvers** → `common/helpers/`. Example:
  `common/helpers/scatter.hpp`'s `scatter_add`, which does a host `+=` /
  device `atomicAdd` and is used by any solver that scatters a face loop into a
  per-cell array. Include it as `"common/helpers/<file>.hpp"`.

## Adding a whole new solver submodule

To bootstrap `<newsolver>` (say `advecDiff`):

1. Create the stage dirs `headers/<newsolver>/`, `common/<newsolver>/`,
   `cpu/<newsolver>/`, `gpu/<newsolver>/` (GPU solvers only),
   `bindings/<newsolver>/`, with `headers/<newsolver>/<newsolver>_compute.hpp`
   (and `.cuh` for a GPU solver) and `bindings/<newsolver>/module.cpp`
   (`NB_MODULE(_core)` calling that solver's `register_*`).
2. Add a `foreach(FBITS ...) foreach(IBITS ...)` target block to `CMakeLists.txt`
   modelled on the advec/advecdiff/utils blocks (GPU-capable:
   `CUDA_ARCHITECTURES`, `CUDA_RUNTIME_LIBRARY Shared`, link `CUDA::cudart`) or,
   for an entirely CPU-only submodule, the `domain` block (`-O3`, no CUDA). Set
   `LIBRARY_OUTPUT_DIRECTORY`/`install DESTINATION` to
   `${pkg}/solvers/<newsolver>` and `OUTPUT_NAME "_core"`.
3. Create `python/manapy_compute_*/solvers/<newsolver>/__init__.py` (×4) that
   re-export from `._core`, and list `<newsolver>` in each
   `python/manapy_compute_*/solvers/__init__.py` docstring.

No `pyproject.toml` change is needed — nested packages under the already-listed
`python/manapy_compute_*` dirs are picked up automatically.

## Verify

There is no checked-in test suite for this module. Verify a kernel by loading
the built `.so` directly
(`importlib.util.spec_from_file_location`/`exec_module`) and exercising it
against an independent reference — a small NumPy reimplementation of the Python
original over a synthetic mesh that hits every branch (for `advec`: interior,
periodic, halo and boundary faces; both 2D and 3D). Every kernel here was
checked this way before being marked done.

## Array types & `ArrayView`

The nanobind↔`ArrayView` glue (`CFVec`/`CFMat`/`FVec`/`CIVec`/`CIMat`/... and
the `D*` device aliases) lives in `src/base/manapy_compute_types.hpp`, shared
with `src/core`/`src/domain`; `make_view<T, N>(arr)` builds an `ArrayView` at
the binding boundary, and `.row(i)`/`.as_const()` are defined once in
`src/base/array_view.hpp`. See [../domain/README.md](../domain/README.md#array-types)
for the full list and for how to add a missing alias or indexing rank.
