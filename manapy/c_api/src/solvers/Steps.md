# Converting `to_convert.py` to C++ — advecdiff solver (CPU + GPU)

Source: [to_convert.py](to_convert.py) — the advection–**diffusion** solver.
This doc plans the port into a new `manapy_compute_<F>_<I>.solvers.advecdiff`
submodule, following [README.md](README.md)'s add-a-kernel recipe (CPU + GPU
variant). Nothing here is implemented yet — boxes are unchecked.

advecdiff is advec plus three deltas: a diffusion (dissipative) flux, a
pluggable numerical flux resolved at compile time, and a diffusion term in the
time step. Everything else mirrors the already-ported `solvers.advec`.

## Function inventory

| `to_convert.py` | becomes | where |
| --- | --- | --- |
| `_explicitscheme_dissipative` | `explicitscheme_dissipative` (CPU+GPU) | `advecdiff` — **new**, diffusion flux |
| `_upwind_flux` / `_centered_flux` / `_rusanov_flux` / `_lax_friedrichs_flux` / `_compute_flux` | `numerical_flux<FluxScheme>` | shared helper in `common/helpers/` |
| `_explicitscheme_convective_2d` | `explicitscheme_convective_2d` (CPU+GPU) | `advecdiff` |
| `_explicitscheme_convective_3d` | `explicitscheme_convective_3d` (CPU+GPU) | `advecdiff` |
| `_time_step` | `time_step` (CPU+GPU) | `advecdiff` — advec's + diffusion term |
| `_update_new_value` | `update_new_value` (CPU+GPU) | **moved to `solvers.utils`** (deduped) |

Names drop the leading underscore, matching the rest of the codebase.

## Decisions

### Flux dispatch: compile-time template + host-side dispatch ✅ decided
Python's `setup(dim, scheme)` binds one flux body to a global `_compute_flux`
so the convective loop calls it with no per-face branch. A `__device__`
function pointer is the wrong translation (indirect call in the hot loop, no
inlining, register pressure). Instead:

- `common/helpers/numerical_flux.hpp` defines
  `enum class FluxScheme { Upwind, Centered, Rusanov, LaxFriedrichs }` and a
  `template <FluxScheme S> MANAPY_COMPUTE_HOST_DEVICE real_t numerical_flux(...)`
  whose body is an `if constexpr` chain (Rusanov / LaxFriedrichs share the
  branch — identical for linear scalar advection).
- The convective element routine and `__global__` kernel take `FluxScheme` as a
  template parameter (on top of `ConvectiveFaceKind`), so the flux inlines with
  zero per-face branches.
- The host `launch_*` (and the CPU driver) do a **single `switch (scheme)`**
  that dispatches to the right instantiation — once per launch, not per face.

Instantiation count: `ConvectiveFaceKind` (4) × `FluxScheme` (4) = 16 kernels
per dim; tiny kernels, not a concern. `order` stays a **uniform runtime
branch** (the order-1 fast path is a memory-footprint win the runtime branch
already achieves; templating it would double instantiations for a single saved
comparison).

`scheme` is passed to the binding as an explicit `index_t` argument (the
dispatch selector) — it replaces Python's global `_compute_flux` binding, since
there is no mutable module-global to bind. SCHEME_IDS = {upwind:0, centered:1,
rusanov:2, lax_friedrichs:3} (2 and 3 route to the same branch).

### Leave advec alone ✅ decided
advec keeps its own runtime-`scheme` `numerical_flux(scheme, …)` in
`common/advec/numerical_flux_common.hpp`; it is **not** retrofitted to the
templated form. Minor, deliberate duplication of the flux math between
`common/advec/` and `common/helpers/` — advecdiff uses the new templated
helper, advec is untouched. (Can be unified later if desired.)

### Dedupe `update_new_value` into `solvers.utils` ✅ decided
`_update_new_value` is identical across solvers, so it moves to `utils`
(`solvers.utils.update_new_value`) and is **removed from `advec`**. advecdiff
does not implement it — it calls the utils one.

**Consequence:** `update_new_value` has a GPU kernel, so `utils` gains CUDA in
Batch 5 — its CMake target gets the arch list, `CUDA::cudart` and
`CUDA_RUNTIME_LIBRARY Shared`, exactly like the advec target, and it becomes a
mixed CPU+GPU module. See the CPU-only-marking convention below for how
`initialisation_gaussian_2d/3d` stay CPU-only inside it.

### CPU-only marking is per-function ✅ decided
A solver module is CPU + GPU by default. A kernel that is **CPU-only** is
marked as such in a comment on its declaration in
`headers/<solver>/<solver>_compute.hpp`; an unmarked kernel is assumed to have
a `launch_<kernel>` GPU counterpart. (Whole-module "CPU-only" — as `utils` and
`domain` are described today — is just the case where every function carries
the mark.) So once `utils` gains CUDA in Batch 5, `initialisation_gaussian_2d/3d`
keep a "CPU-only" comment on their declarations while `update_new_value` does
not.

### New submodule `solvers.advecdiff` ✅ decided
Its own `_advecdiffcore_<F>_<I>` CMake target, GPU-capable like advec, output
to `${pkg}/solvers/advecdiff`. Shares the single `bindings/registry.hpp`.

## Batches

One batch = one self-contained, independently-verifiable chunk, ordered so no
batch depends on a later one. Verify each **end-to-end, not just compiled**:
load the built `.so` (or a standalone driver) and compare against an
independent NumPy transcription of the Python original over a synthetic mesh
that exercises every branch (interior / periodic / halo / boundary faces; 2D
and 3D; every `FluxScheme`; order 1 and order 2).

### Batch 0 — shared flux helper (no bindings) ✅ done
`common/helpers/numerical_flux.hpp`: `FluxScheme` enum + templated
`numerical_flux<S>()`. Solver-agnostic, no Python binding (an internal helper).
Written first — Batch 2 includes it. Verified with a standalone driver
(upwind=3.0, centered=4.5, rusanov=3.0 for w_l=1, w_r=2, sign=3), matching the
Python bodies.

- [x] `FluxScheme` enum
- [x] `numerical_flux<FluxScheme>` (upwind / centered / rusanov==lax_friedrichs)

### Batch 1 — bootstrap the `advecdiff` submodule ✅ done
Empty module that builds and imports, so later batches only add kernels.
Verified: `_advecdiffcore_64_64` builds and `from
manapy_compute_64_64.solvers import advecdiff` imports (empty `__all__`, correct
module docstring).

- [x] `headers/advecdiff/advecdiff_compute.hpp` + `.cuh` (decl headers; note the
      per-function CPU-only-marking convention in the .hpp)
- [x] `bindings/advecdiff/module.cpp` (`NB_MODULE(_core)`, no registrations yet)
- [x] `CMakeLists.txt`: `_advecdiffcore_<F>_<I>` target block (GPU-capable,
      modelled on advec), include dirs `src/base src/solvers
      src/solvers/headers/advecdiff`, install to `${pkg}/solvers/advecdiff`
- [x] `python/manapy_compute_*/solvers/advecdiff/__init__.py` (×4); listed
      `advecdiff` in each `solvers/__init__.py` docstring and the top-level one

### Batch 2 — convective residual (2D + 3D) ✅ done
Mirrors advec's convective kernels, minus the runtime `scheme` (now a template)
and plus the order-1 fast path. Verified end-to-end against a NumPy transcription
on synthetic 2D/3D meshes hitting all four face kinds (incl. periodic 11/33 in
2D and 11/33/55 in 3D): **all 4 schemes × {order 1, order 2}** match on the CPU,
and a GPU on-device spot check (2D, rusanov, order 2, MX550) matches too.

- [x] `common/advecdiff/convective_common.hpp` — advecdiff's own
      `ConvectiveFaceKind` enum + includes of `common/helpers/scatter.hpp` and
      `common/helpers/numerical_flux.hpp` (advec left untouched)
- [x] `common/advecdiff/explicitscheme_convective_2d_common.hpp` — templated on
      `<ConvectiveFaceKind, FluxScheme>`, order-1 fast path
- [x] `common/advecdiff/explicitscheme_convective_3d_common.hpp`
- [x] `cpu/advecdiff/*_cpu.cpp` (×2): zero `rez_w`, order-1 fast path
      (reads only `w_c`/`U.face`/normal) vs generic path, `switch(scheme)`
      dispatch, sweep the four face lists
- [x] `gpu/advecdiff/*_cuda.cu` (×2): `switch(scheme)` in `launch_*` around the
      per-face-kind kernels; atomicAdd scatter
- [x] decls in `advecdiff_compute.hpp`/`.cuh`; binding (`_cuda` twin, explicit
      `scheme` arg); registry/module/CMake/`__init__.py` wiring. Register
      functions are prefixed `register_advecdiff_*` to avoid clashing with
      advec's same-named ones in the shared `bindings/registry.hpp`.
- [x] verify: all 4 face kinds × 4 schemes × {order 1, order 2}, 2D and 3D,
      against NumPy (order-1 path is bit-identical to the generic path)

### Batch 3 — dissipative (diffusion) flux ✅ done
New in advecdiff. One dimension-agnostic kernel over **all** faces (reads all 3
normal components + `wz_face`; no `d_*face` lists are passed). Per-face runtime
branch on `face_name == 0` (interior → scatter to both cells, else → left only)
— the divergence is just one extra `scatter_add`. Note the scatter sign is the
opposite of the convective residual's (owner +q, interior neighbour -q).
Verified against NumPy on a 4-cell/6-face mesh mixing 3 interior and 3 boundary
faces with non-zero Dxx/Dyy/Dzz: CPU matches for both 3D (z components set) and
2D (wz_face and normal z zeroed), and a GPU on-device run (MX550) matches too.

- [x] `common/advecdiff/dissipative_common.hpp` — `dissipative_face` (HOST_DEVICE),
      reuses `scatter_add`
- [x] `cpu/advecdiff/explicitscheme_dissipative_cpu.cpp` (zero `dissip_w`, loop)
- [x] `gpu/advecdiff/explicitscheme_dissipative_cuda.cu` (zero kernel + face
      kernel + `launch_*`)
- [x] decls + binding (`_cuda` twin) + wiring (`register_advecdiff_*` prefix)
- [x] verify: mixed interior/boundary faces, non-zero `Dxx/Dyy/Dzz`, 2D and 3D

### Batch 4 — time step (with diffusion) ✅ done
advec's `time_step` reduction plus the diffusion term
`lam += (Dxx+Dyy+Dzz)·mes² / cell_volume`, `mes = ‖face_normal‖` (needs a
portable `sqrt`, host+device). Extra `Dxx/Dyy/Dzz` args; `face_measure`/`dim`
still unused (kept for signature parity). Same atomicMin reduction on the GPU
as advec's. A `time_step_sqrt` host/device helper keeps `mes` bit-faithful to
NumPy (the `mes²` cancels the root but the rounding is preserved). Verified
against a NumPy transcription on a 6-cell mesh: CPU matches **bit-exactly**
(atol=0) with and without diffusion, and `lam==0` (u=v=w=0, D=0) returns the
1e6 seed; a standalone on-device GPU driver (MX550, sm_75) matches bit-exactly
too via the atomicMin bit-trick.

- [x] `common/advecdiff/time_step_common.hpp` — per-cell candidate incl. diffusion
- [x] `cpu/advecdiff/time_step_cpu.cpp` + `gpu/advecdiff/time_step_cuda.cu`
- [x] decls + binding (`_cuda` twin, GPU returns the scalar) + wiring
- [x] verify: cells with/without diffusion, `lam==0` skip, against NumPy

### Batch 5 — dedupe `update_new_value` into `utils` ✅ done
Moved the identical `_update_new_value` out of advec into `solvers.utils`,
making utils a mixed CPU + GPU module (per the per-function marking convention:
`initialisation_gaussian_*` stay CPU-only, `update_new_value` gains a GPU twin).
Verified: `_adveccore_64_64` and `_solversutils_64_64` both build; advec no
longer exposes `update_new_value` while `utils` now exposes it and its `_cuda`
twin; utils' CPU `update_new_value` is bit-identical to the NumPy reference; the
on-device GPU driver (MX550, sm_75) matches to 1 ULP (the standard FMA
contraction — bit-identical under `--fmad=false`, same as advec's original
kernel, whose code was moved verbatim).

- [x] Add `update_new_value` (CPU+GPU) to `utils`: `utils_compute.hpp` + new
      `utils_compute.cuh`, `common/utils/`, `cpu/utils/`, **new**
      `gpu/utils/update_new_value_cuda.cu`, `bindings/utils/` (`_cuda` twin),
      registry/module wiring, `__init__.py`
- [x] Make the `_solversutils_<F>_<I>` CMake target GPU-capable (CUDA arch,
      `CUDA::cudart`, `CUDA_RUNTIME_LIBRARY Shared`)
- [x] Remove `update_new_value` from `advec` (common/cpu/gpu/binding/registry/
      module/CMake/`__init__.py`)
- [x] Update README + `solvers/__init__.py` docstrings (utils no longer CPU-only)
- [x] verify: utils' `update_new_value` matches the old advec one; advec still
      builds/imports without it

## Module bootstrap note

Like advec (see [README.md](README.md#adding-a-whole-new-solver-submodule)),
advecdiff registers into its **own** `_core` extension, not advec's — its own
`bindings/advecdiff/module.cpp` and `_advecdiffcore_<F>_<I>` target, sharing the
solvers-level `bindings/registry.hpp`. No `pyproject.toml` change needed
(nested packages under the listed `python/manapy_compute_*` dirs are picked up
automatically).
