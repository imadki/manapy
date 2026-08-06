# Adding a new kernel `<kernel>` to manapy.domain

CPU-only: there's no `gpu/` counterpart here, so a kernel's whole
implementation lives in one `.cpp` file — no `common/<kernel>_common.hpp`
split the way [../core/README.md](../core/README.md)'s recipe uses. This
module also registers into its own nanobind module, separate from
`src/core`'s (own `bindings/registry.hpp`/`module.cpp`, own
`_domaincore_<F>_<I>` CMake targets), but ships inside the same
`manapy_compute_<F>_<I>` Python package as `src/core`, as its `domain`
submodule (`src/core` is the `core` submodule).

1. Declare it in `domain_compute.hpp`
   - `void <kernel>(ArrayView<...> ..., ...);` — `ArrayView` args in the
     same order as the Python original, `const` for read-only ones
   - a function only ever called by other kernels here, never a standalone
     Python entry point, doesn't belong here — see "Internal-only helpers"
     below instead

2. Create `cpu/<kernel>_cpu.cpp`
   - `#include "domain_compute.hpp"`
   - `#include "common/domain_helpers.hpp"` if it needs a shared helper
   - define `void <kernel>(...)` (or `index_t <kernel>(...)` if it returns
     a scalar), looping over elements directly — no `_element` split, no
     GPU counterpart to share code with
   - a Python `raise RuntimeError(...)` becomes `throw
     std::runtime_error(...)`; nanobind turns that into a Python
     `RuntimeError` automatically at the binding boundary

3. Create `bindings/<kernel>.cpp`
   - `#include "manapy_compute_types.hpp"`, `"bindings/registry.hpp"`,
     `"domain_compute.hpp"`
   - define `<kernel>_py(...)` taking `CFVec`/`CFMat`/`FVec`/`FMat`/
     `CIVec`/`CIMat`/`IVec`/`IMat`/`CI8Vec`/`I8Vec`/`FTensor` args (see
     "Array types" below), calling `<kernel>(...)` via `make_view<...>`
   - define `void register_<kernel>(nb::module_ &m)` with a single
     `m.def("<kernel>", ...)` — no `_cuda` twin
   - mark every **output** (mutable) argument `.noconvert()`, e.g.
     `nb::arg("res").noconvert()`. Without it, a caller passing an array of
     the wrong dtype gets no error and no results: nanobind casts it to a
     temporary, the kernel fills the temporary, and the caller's array comes
     back untouched. Read-only inputs should stay convertible — the silent
     cast is a convenience there, and matches what the c_api's
     `PyArray_FROM_OTF` did. Every existing binding in the project already
     carries this; a new one without it is the odd one out.

4. Edit `bindings/registry.hpp`
   - add `void register_<kernel>(nb::module_ &m);`

5. Edit `bindings/module.cpp`
   - add `register_<kernel>(m);` inside `NB_MODULE(_core, m)`

6. Edit `CMakeLists.txt`
   - add `src/domain/bindings/<kernel>.cpp` and
     `src/domain/cpu/<kernel>_cpu.cpp` to the `nanobind_add_module(...)`
     source list inside the `_domaincore_<F>_<I>` `foreach` block (the
     second one in the file — the first is `src/core`'s `_gradcore_<F>_<I>`)

7. Edit all four `python/manapy_compute_*/domain/__init__.py`
   - add `<kernel>` to the `from ._core import (...)` block and to
     `__all__` (keep both alphabetically sorted)

8. Rebuild
   - `cmake --build build/vscode --target _domaincore_32_32 _domaincore_32_64 _domaincore_64_32 _domaincore_64_64`

9. Verify, not just compile
   - load the built `.so` directly (`importlib.util.spec_from_file_location`,
     `spec.loader.exec_module`) and exercise the new function against
     hand-built or independently-computed values. There's no checked-in
     test suite for this module yet — every kernel here was verified this
     way, ad hoc, before being marked done in [Steps.md](Steps.md) (which
     also records the specific scenario used for each one).

10. Reinstall if not already editable
    - `pip install -e . --no-build-isolation`

## Internal-only helpers

A function only ever called by other kernels in this module — never a
standalone Python entry point — skips steps 1, 3, 4, 5, 7 above. Add it as
a plain `inline` function directly to `common/domain_helpers.hpp` instead
(no Python binding), and `#include "common/domain_helpers.hpp"` from
whichever `cpu/<kernel>_cpu.cpp` needs it.

## Array types

`src/base/manapy_compute_types.hpp` holds the nanobind↔`ArrayView` glue
shared with `src/core`. Check it doesn't already have what you need before
adding a new alias:

- `CFVec`/`CFMat` — const `real_t`, 1D/2D (shared with `src/core`)
- `FVec`/`FMat` — mutable `real_t`, 1D/2D (`FMat` added for this module —
  `src/core` only ever writes 1D float outputs)
- `FTensor` — mutable `real_t`, 3D (added for `create_normal_face_of_cell`'s
  `cell_nf`; the only rank-3 array either module uses)
- `CIVec`/`CIMat` — const `index_t`, 1D/2D (shared with `src/core`)
- `IVec`/`IMat` — mutable `index_t`, 1D/2D (added for this module —
  `src/core` never writes an int array)
- `CI8Vec`/`I8Vec` — const/mutable `int8_t`, 1D (`to_convert.py`'s
  `cell_type`, `b_visited`)

If a kernel needs a shape `ArrayView` doesn't support yet — there's no
generic N-dimensional indexing, only what's actually been needed (1D, 2D,
3D) — add the matching `operator()(i, j, ...)` overload to
`src/base/array_view.hpp`, gated the same way the existing ones are
(`requires(NDIM == <n>)`).

## `ArrayView` conveniences

Defined once in `array_view.hpp`; apply to any `ArrayView`, `src/core`'s
included:

- `.row(i)` — view row `i` of a 2D array as a 1D one (respects stride).
  Lets a function typed for `ArrayView<T, 1>` run directly on one row of a
  2D array (e.g. `node_cellid.row(node)`) without copying it out.
- `.as_const()` — read-only view over the same data. Needed when a mutable
  buffer (e.g. `some_matrix.row(i)`) has to be passed to a function whose
  signature only reads its argument (`ArrayView<const T, NDIM>`).

## Parallelism (OpenMP)

`CMakeLists.txt` has an optional `find_package(OpenMP COMPONENTS CXX)`,
linked into every `_domaincore_<F>_<I>` target when found (harmless if
not: an unrecognized `#pragma omp` just runs sequentially). Only
`create_cellfid` uses it today. Before adding `#pragma omp parallel for` to
a new kernel's outer loop, confirm each iteration is actually independent:
- it only writes to output rows/entries keyed by the loop variable (never
  another cell/node/face's row — several kernels here, e.g.
  `create_cell_cellnid`, `create_ghost_tables`, deliberately write into a
  *different* row than the one being iterated, which is **not** safe to
  parallelize this way without extra care)
- any scratch it needs is declared fresh inside the loop body, not a
  shared buffer passed in from outside (a shared scratch buffer, like
  `create_info`'s `tmp_cell_faces`, is exactly what makes a loop *not*
  parallelizable this way)
