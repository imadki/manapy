# Porting `manapy/c_api` (METIS partitioning) into manapy_compute

All batches (0–7) are implemented. §11 records the deviations and pre-existing
c_api defects found along the way, and supersedes §1, §4 and §5 in places.

Source: `../new_manapy/manapy/c_api` (Python C API + NumPy C API, vendored
METIS/GKlib, 4 standalone extension modules).
Target: a new `partitioning` submodule of the existing
`manapy_compute_<float bits>_<int bits>` packages, bound with **nanobind**,
**CPU only**, with METIS/GKlib compiled from the vendored sources.

---

## 0. What is being ported

| c_api file | lines | Python/NumPy C API usage | port difficulty |
|---|---|---|---|
| `includes/Types.h` | 40 | `NPY_*` type codes, `MODULE_NAME` | replaced wholesale |
| `includes/PyArray.h` | 242 | **heavy** — owns `PyArrayObject`, allocates via `PyArray_SimpleNew`/`PyArray_ZEROS` | replaced wholesale |
| `includes/manapy_part.h` | 85 | declarations only | mechanical |
| `includes/LocalDomainStruct.h` + `.cpp` | 158 + 81 | holds 18 `PyArray*`, `Py_BuildValue` 27-tuple | rewrite of storage + tuple build |
| `src/utils.cpp` | 161 | `PyImport_ImportModule("sys")` for `print_instant` | small |
| `src/compute_cell_center_volume.cpp` | 224 | none beyond `PyArray` accessors | mechanical |
| `src/partitioning.cpp` | 1256 | ~25 `new PyArray<...>` allocations, `PyList_New`/`PyList_SET_ITEM` | mechanical but bulk of the work |
| `src/py_manapy_part.cpp` | 823 | **all of it** — `PyArg_ParseTuple`, `PyArray_FROM_OTF`, `PyMethodDef`, `PyInit_*` | rewritten as nanobind bindings |

Public surface = 6 functions:
`make_n_part_graph_k_way`, `make_n_part_mesh_dual`, `make_n_part_mesh_nodal`,
`create_local_domains`, `compute_cell_center_area_2d`,
`compute_cell_center_volume_3d`.

---

## 1. Target layout

The instruction is to keep `c_api/src`'s structure, so this module deliberately
does **not** follow the one-kernel-per-file `bindings/` + `cpu/` + `common/`
split used by `src/core`, `src/domain`, `src/boundary` and `src/solvers`. It
keeps c_api's coarse-grained files instead:

```
third_party/CMakeLists.txt                 metis_i{32,64}_f{32,64} static libs
third_party/GKlib/                         vendored (copied from c_api)
third_party/METIS/                         vendored (copied from c_api)

src/partitioning/Steps.md                  this file
src/partitioning/README.md                 module docs (mirrors c_api/README*.md)
src/partitioning/includes/manapy_part.hpp  CELL_TYPE enum + all decls (was manapy_part.h)
src/partitioning/includes/local_domain_struct.hpp
src/partitioning/src/partitioning.cpp      same content, PyArray -> OwnedArray/ArrayView
src/partitioning/src/utils.cpp
src/partitioning/src/local_domain_struct.cpp
src/partitioning/src/compute_cell_center_volume.cpp
src/partitioning/bindings/registry.hpp     register_* decls (project convention)
src/partitioning/bindings/module.cpp       NB_MODULE(_core)
src/partitioning/bindings/metis_partition.cpp   the 3 make_n_part_* bindings
src/partitioning/bindings/create_local_domains.cpp
src/partitioning/bindings/cell_center_volume.cpp

src/base/owned_array.hpp                   NEW — shared owning-array helper (see §4)

python/manapy_compute_<F>_<I>/partitioning/__init__.py   x4
```

`includes/Types.h` and `includes/PyArray.h` disappear: their jobs are taken over
by the existing `src/base/precision.hpp` / `src/base/manapy_compute_types.hpp`
and the new `src/base/owned_array.hpp`.

Submodule name: `partitioning` (matches `manapy.domain.Partitioning` used in
`main.py`). Alternative `part` is shorter but less descriptive.

---

## 2. Precision mapping — note the transposed naming

c_api names modules `manapy_part<INT bits>_<FLOAT bits>`
(`create_manapy_module(manapy_part32_64 32 64 metis_32)` = int32 + float64).
This repo names packages `manapy_compute_<FLOAT bits>_<INT bits>`. **The two
conventions are reversed.** The mapping to implement:

| c_api module | this repo |
|---|---|
| `manapy_part32_32` | `manapy_compute_32_32.partitioning` |
| `manapy_part32_64` | `manapy_compute_64_32.partitioning` |
| `manapy_part64_32` | `manapy_compute_32_64.partitioning` |
| `manapy_part64_64` | `manapy_compute_64_64.partitioning` |

Type aliases:

- `fdx_t` → `real_t` (from `src/base/precision.hpp`)
- `idx_t` → METIS's own `idx_t`; add
  `static_assert(std::is_same_v<idx_t, index_t>)` in `manapy_part.hpp` so a
  mismatched `IDXTYPEWIDTH` fails at compile time rather than silently
  corrupting data.
- `NPY_INT_TYPE` / `NPY_FLOAT_TYPE` → gone; nanobind infers dtype from `T`.
- `MODULE_NAME` / `STR()` → gone; every target builds `_core` and gets its
  identity from its install directory, exactly like `core`/`domain`/`boundary`.

### 2.1 `real_t` collision with METIS — the reason for 4 third-party builds

`METIS/include/metis.h` defines a **global** `typedef float real_t;` (or
`double`, per `REALTYPEWIDTH`). `src/base/precision.hpp` defines a global
`using real_t = ...`. Any TU that includes both — which is every partitioning
TU — only compiles if the two widths agree; otherwise it is a hard
"conflicting declaration" error. c_api never hit this because it never included
`precision.hpp`.

Therefore METIS/GKlib must be built **four times**, one per (int, float) pair —
which is also what the request ("compile the third party libraries
(intxx/floatxx)") asks for:

| target | `IDXTYPEWIDTH` | `REALTYPEWIDTH` | used by |
|---|---|---|---|
| `metis_i32_f32` | 32 | 32 | `manapy_compute_32_32` |
| `metis_i32_f64` | 32 | 64 | `manapy_compute_64_32` |
| `metis_i64_f32` | 64 | 32 | `manapy_compute_32_64` |
| `metis_i64_f64` | 64 | 64 | `manapy_compute_64_64` |

(METIS's `real_t` only shows up in `tpwgts`/`ubvec`, which we always pass as
`nullptr`, so widening it is behaviourally inert.)

The rejected alternative — isolating `metis.h` behind a thin C shim TU that
never sees `precision.hpp` — would keep 2 static libs but scatter the METIS
calls across an extra indirection layer, and `manapy_part.hpp`/
`local_domain_struct.hpp` use `idx_t` throughout. Not worth it.

---

## 3. Third-party build

New `third_party/CMakeLists.txt`, modelled on c_api's but parameterised:

```cmake
file(GLOB GKLIB_SOURCES  "${CMAKE_CURRENT_SOURCE_DIR}/GKlib/src/*.c")
file(GLOB METIS_SOURCES  "${CMAKE_CURRENT_SOURCE_DIR}/METIS/libmetis/*.c")

function(add_metis_variant tgt ibits fbits)
  add_library(${tgt} STATIC ${GKLIB_SOURCES} ${METIS_SOURCES})
  target_compile_definitions(${tgt} PUBLIC IDXTYPEWIDTH=${ibits} REALTYPEWIDTH=${fbits})
  target_include_directories(${tgt} PUBLIC
      GKlib/include METIS/include METIS/libmetis)
  set_target_properties(${tgt} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden)     # keep METIS symbols out of the .so dynamic table
endfunction()
```

Added from the top-level `CMakeLists.txt` with
`add_subdirectory(third_party EXCLUDE_FROM_ALL)`.

Notes / checks:

- Only `libmetis/*.c` is globbed — METIS's `programs/` (gpmetis CLI) and
  `utils/` are not built, matching c_api.
- Static libs, `POSITION_INDEPENDENT_CODE ON`, so they link into each `_core`
  `.so`. Nothing is exported: Python `dlopen`s extensions `RTLD_LOCAL`, and
  `C_VISIBILITY_PRESET hidden` means importing e.g. `manapy_compute_32_32` and
  `manapy_compute_64_64` in one process cannot cross-bind their two different
  `METIS_PartGraphKway`s.
- Vendoring: copy `GKlib/` and `METIS/` verbatim (55 MB as-is). Prune
  `METIS/graphs/`, `METIS/manual/`, `METIS/programs/`, `GKlib/apps/`,
  `GKlib/test/` before committing — none are compiled, and it cuts the bulk of
  the size. Keep both `LICENSE` files (Apache-2.0) and add a
  `third_party/README.md` recording upstream URL + commit.
- Warning noise: GKlib/METIS are C89-ish and warn a lot under the project's
  C++20/`-O3` flags. Build them as C (they are `.c`), and don't add `-Wall`.

---

## 4. Replacing the NumPy C API

`PyArray<T, Dim>` does **two** distinct jobs; each maps to a different
replacement.

### 4.1 Inputs — already solved by the project

`PyArray_FROM_OTF(obj, NPY_INT_TYPE, NPY_ARRAY_IN_ARRAY)` + `PyArray<idx_t,2>`
wrapping becomes the existing pattern from `src/base/manapy_compute_types.hpp`:
typed `nb::ndarray` alias in the binding signature, `make_view<T, N>()` to get
an `ArrayView<T, N>`. Existing aliases cover every input:

| c_api input | alias |
|---|---|
| `part_vert`, `phy_faces_name` (const 1D int) | `CIVec` |
| `node_cellid`, `node_phyid`, `cells`, `phy_faces`, `graph` (const 2D int) | `CIMat` |
| `cells_type` (const 1D int8) | `CI8Vec` |
| `nodes` (const 2D float) | `CFMat` |
| `cell_area`/`cell_volume` (mutable 1D float, in-place out) | `FVec` |
| `cell_center` (mutable 2D float, in-place out) | `FMat` |

No new aliases needed. **Verify early** (Batch 1 spike) that nanobind performs
the implicit dtype/contiguity conversion that `PyArray_FROM_OTF` used to do
silently — the rest of the repo already relies on this, but partitioning is the
first place a caller is likely to pass a `float64` array to a `float32` build.
If conversion is not automatic for some case, add an explicit
`np.ascontiguousarray(..., dtype=...)` in the Python wrapper rather than
loosening the C++ signature.

Accessor rewrite inside the ported `.cpp` files:

| `PyArray` | `ArrayView` |
|---|---|
| `a->get(i)` | `a(i)` |
| `a->get2(i, j)` | `a(i, j)` |
| `a->shape[d]` | `a.size(d)` |
| `a->last()` | `a(a.size(0) - 1)` |
| `a->last2(i)` | `a(i, a.size(1) - 1)` |
| `a->sub_array(i)` | `a.row(i)` |

`ArrayView` already has `row()` and `as_const()`, so `utils.cpp`'s
`binary_search(const PyArray<idx_t,1>&, idx_t)` and `intersect_arr` port to
`ArrayView<const index_t, 1>` / `<const index_t, 2>` with no logic change.

### 4.2 Outputs — new `src/base/owned_array.hpp`

Partitioning is the first module in this repo whose output sizes are not known
to the caller (`nb_halos`, `l_map_phy_faces.size()`, …), so it cannot use the
"caller preallocates, kernel fills in place" convention that `core`/`domain`
use. It must allocate and return. That is a deliberate, documented deviation —
`compute_cell_center_*` keeps the in-place convention because its sizes *are*
known.

Sketch:

```cpp
template <typename T, int NDIM>
class OwnedArray {
  std::unique_ptr<T[]> buf_;
  size_t shape_[NDIM];
public:
  OwnedArray() = default;                  // empty: no buffer, zero extents
  OwnedArray(std::array<size_t, NDIM> shape, bool zero_init = false);
  ArrayView<T, NDIM> view() const;         // hands the existing accessors back
  T &operator()(...);                      // same call syntax as ArrayView
  explicit operator bool() const;          // holds a buffer?
  void reset();                            // back to empty
  nb::ndarray<nb::numpy, T> release();     // transfers ownership to Python
};
```

`release()` detaches the buffer and attaches an `nb::capsule` deleter, so NumPy
frees it — the nanobind equivalent of
`PyArray_SimpleNewFromData` + `PyArray_ENABLEFLAGS(NPY_ARRAY_OWNDATA)`:

```cpp
T *p = buf_.release();
nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<T *>(q); });
return nb::ndarray<nb::numpy, T>(p, NDIM, shape_, owner);
```

Same trick for the raw `malloc`'d METIS output (`part_idx` in
`make_n_part_*`): capsule with a `free` deleter, zero copies, matching c_api.

Constructor maps directly onto c_api's two allocation modes
(`PyArray_SimpleNew` → `zero_init=false`, `PyArray_ZEROS` → `true`); the one
zero-initialised allocation today is `ld[i].nodes` in `loop_through_nodes`, and
that must be preserved.

`make_npy_dims(...)` → `std::array<size_t, N>{...}` (or keep a
`make_dims(...)` helper with the same call sites, to keep the diff small).

### 4.3 `LocalDomainStruct`

- 18 `PyArray<T,D>*` members → `OwnedArray<T,D>` **held by value**. The mapping
  Batch 4 needs:

  | c_api | now |
  |---|---|
  | `x = new PyArray<idx_t,2>(make_npy_dims(r, c))` | `x = OwnedArray<index_t,2>(make_dims(r, c))` |
  | `new PyArray<...>(dims, true)` (zeroed) | `x = OwnedArray<...>(make_dims(...), true)` |
  | `if (x == nullptr)` | `if (!x)` |
  | `x->shape[d]` | `x.size(d)` |
  | `x->get(i)` / `x->get2(i, j)` | `x(i)` / `x(i, j)` |
  | `delete x; x = nullptr;` | `x.reset()` |

  By value rather than behind `optional` or `unique_ptr` because `OwnedArray`
  is default-constructible into an empty state, so "not allocated yet" needs no
  wrapper. A wrapper would add a *second* emptiness flag next to `OwnedArray`'s
  own, and the two can disagree — `release()` empties the array without
  disengaging anything around it, so an `optional` would still test as engaged
  for a table whose buffer is gone. `unique_ptr` would additionally heap-
  allocate a control block for something whose lifetime is exactly the struct's.
  Holding by value also keeps the call sites closest to the c_api original
  (`x(i, j)` rather than `(*x)(i, j)`), which matters across
  `partitioning.cpp`'s 1256 lines.

  For a hot loop, bind the view once —
  `auto centvol = ld[p].halo_centvol.view();` then `centvol(row, 0)` — rather
  than going through `operator()` each time; that is the idiom the rest of the
  project uses.
- `PyObject *tuple_res` → `nb::object tuple_res` (nanobind refcounts it; the
  hand-written destructor's `Py_XDECREF` disappears).
- `create_tuple()` → `nb::make_tuple(...)` over the same 27 items in the same
  order: 18 arrays via `release()`, then the 9 `max_*` scalars.
  `Py_BuildValue`'s `"i"` format truncated `idx_t` to C `int`; nanobind casts
  `index_t` correctly — a latent int64 bug that disappears.
- `free_tables()` → `.reset()` on each member (or drop it entirely; the members
  are moved out by `release()` and destructors handle the rest).
- `get_result_as_py_list` (`PyList_New`/`PyList_SET_ITEM`) → `nb::list` +
  `append`.

### 4.4 Error handling

Delete every `PyErr_SetString` / `PyErr_Format` / `return nullptr` path and
throw instead; nanobind translates:

| c_api | replacement |
|---|---|
| `PyErr_SetString(PyExc_ValueError, ...)` | `throw std::invalid_argument(...)` → `ValueError` |
| `PyErr_SetString(PyExc_MemoryError, ...)` | `throw std::bad_alloc()` → `MemoryError` |
| `PyErr_Format(PyExc_RuntimeError, "METIS_... (status=%d)", ret)` | `throw std::runtime_error(...)` → `RuntimeError` |

This removes the manual `free_tables()` lambdas and every `Py_XDECREF` unwind
path in `py_manapy_part.cpp` — RAII (`std::unique_ptr`, `OwnedArray`) covers
them. The `try/catch` around `create_sub_domains` in
`py_create_local_domains` also goes away.

### 4.5 `utils.cpp` — `print_instant`

`PyImport_ImportModule("sys")` + `PyObject_GetAttrString` +
`PyObject_CallMethod(stdout, "write"/"flush")` becomes:

```cpp
nb::module_::import_("sys").attr("stdout").attr("write")(str);
```

with the same `manapy_debug_timing_enabled()` env-var gate and the same
`DEBUG_PRINT_INSTANT` / `DEBUG_TIME_IT` macros.

### 4.6 GIL

Keep the GIL held throughout, as c_api did. `create_local_domains` allocates
Python-owned arrays and `print_instant` touches `sys.stdout` from deep inside
the pipeline, so a blanket `nb::gil_scoped_release` is not safe. If the ~seconds
of `loop_through_*` on large meshes turns out to matter, release the GIL around
those *specific* phases later — out of scope for this port.

---

## 5. Binding surface

`bindings/module.cpp` defines `NB_MODULE(_core, m)` and calls the `register_*`
functions declared in `bindings/registry.hpp`, exactly like
`src/domain/bindings/module.cpp`.

```cpp
// bindings/metis_partition.cpp
IVec-returning:
  make_n_part_graph_k_way(graph: CIMat, nb_part: index_t)      -> ndarray[index_t, 1]
  make_n_part_mesh_dual  (cells: CIMat, nb_parts: index_t,
                          n_common: index_t)                    -> ndarray[index_t, 1]
  make_n_part_mesh_nodal (cells: CIMat, nb_parts: index_t)      -> ndarray[index_t, 1]

// bindings/create_local_domains.cpp
  create_local_domains(part_vert: CIVec, node_cellid: CIMat, node_phyid: CIMat,
                       cells: CIMat, cells_type: CI8Vec, nodes: CFMat,
                       phy_faces: CIMat, phy_faces_name: CIVec,
                       nb_parts: index_t, dim: index_t)         -> list[tuple]

// bindings/cell_center_volume.cpp
  compute_cell_center_area_2d  (cells: CIMat, nodes: CFMat,
                                cell_area: FVec,   cell_center: FMat) -> None
  compute_cell_center_volume_3d(cells: CIMat, nodes: CFMat,
                                cell_volume: FVec, cell_center: FMat) -> None
```

All with `nb::arg("...")` names matching c_api's positional order, so existing
callers keep working.

Behaviour changes to call out:

1. `compute_cell_center_*` returned `PyFloat_FromDouble(Py_NAN)` in c_api. They
   return `None` here. `manapy_c_api.py` already discards the value.
2. `create_local_domains`' docstring in c_api advertises a **22**-item tuple,
   but `LocalDomainStruct::create_tuple` builds **27** (18 arrays + 9 `max_*`
   scalars: `max_cell_nodeid`, `max_cell_faceid`, `max_face_nodeid`,
   `max_node_haloid`, `max_cell_halonid`, `max_node_phyid`,
   `max_node_halophyid`, `max_cell_phyid`, `max_cell_halophyid`). The docstring
   is stale — write the correct 27-item list in the new binding and in
   `partitioning/__init__.py`.
3. `nb_parts < 2` and `dim not in (2, 3)` validation is preserved verbatim.

---

## 6. CMake

Top-level `CMakeLists.txt`, after the existing `domain` block, add one more
`foreach(FBITS) foreach(IBITS)` block — CPU-only, so shaped like the `domain`
target (no `CUDA_ARCHITECTURES`, no `CUDA::cudart`):

```cmake
add_subdirectory(third_party EXCLUDE_FROM_ALL)   # once, before the loops

foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _partitioningcore_${FBITS}_${IBITS})
    nanobind_add_module(${tgt} NB_STATIC
      src/partitioning/bindings/module.cpp
      src/partitioning/bindings/metis_partition.cpp
      src/partitioning/bindings/create_local_domains.cpp
      src/partitioning/bindings/cell_center_volume.cpp
      src/partitioning/src/partitioning.cpp
      src/partitioning/src/utils.cpp
      src/partitioning/src/local_domain_struct.cpp
      src/partitioning/src/compute_cell_center_volume.cpp)
    target_include_directories(${tgt} PRIVATE
      src/base src/partitioning src/partitioning/includes)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS} MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE -O3)
    target_link_libraries(${tgt} PRIVATE metis_i${IBITS}_f${FBITS})
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/partitioning")
    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/partitioning)
  endforeach()
endforeach()
```

`metis_i${IBITS}_f${FBITS}` is the naming that makes the transposition of §2
disappear at the CMake level: `IBITS` picks `IDXTYPEWIDTH`, `FBITS` picks
`REALTYPEWIDTH`.

Also: the project is currently `LANGUAGES CXX CUDA`; METIS/GKlib are C, so add
`C` → `project(manapy_compute LANGUAGES C CXX CUDA)`.

`c_api/CMakeLists.txt`'s NumPy-include lookup and its `python_add_library` /
`PREFIX ""` handling are all dropped — `nanobind_add_module` covers it, and
nanobind needs no NumPy headers.

---

## 7. Python packaging

- Four new `python/manapy_compute_<F>_<I>/partitioning/__init__.py`, in the
  style of the existing `domain/__init__.py`: explicit
  `from ._core import (...)` + `__all__`, with the corrected 27-item tuple
  documentation.
- Add `"partitioning"` to `_SUBMODULES` in each package's top-level
  `__init__.py` (lazy PEP-562 loading already handles the rest) and mention it
  in the module docstring.
- `pyproject.toml`: `wheel.packages` already lists the four `python/...`
  package roots, and subpackages come along — **verify** with
  `pip wheel . && unzip -l` that `partitioning/__init__.py` and
  `partitioning/_core*.so` both land in the wheel; if not, list them
  explicitly. No new build requirement (nanobind ≠ NumPy C API), so c_api's
  `requires = [..., "numpy"]` is not carried over.
- `README.md`: extend the Layout section and the intro (it still describes the
  project as "2D cell-gradient kernels").
- `.gitignore`: nothing needed; `third_party/` is committed on purpose. Confirm
  the blanket `*.so` rule doesn't swallow anything vendored (it should not —
  GKlib/METIS ship sources only).
- Downstream note (in `../new_manapy`, not this repo): `manapy/c_api/
  manapy_c_api.py` builds `api_dic[INT_TYPE][FLOAT_TYPE]` from
  `manapy_part<INT>_<FLOAT>`. Repointing it at
  `manapy_compute_<FLOAT>_<INT>.partitioning` must respect the transposition in
  §2 — the single most likely place to introduce a silent precision mismatch.

---

## 8. Build order (batches)

Each batch ends with a working `pip install .`.

- **Batch 0 — third party. DONE.** Vendored + pruned GKlib/METIS (28 MB, down
  from 55), wrote `third_party/CMakeLists.txt`, added `C` to `project()`.
  All four `metis_i*_f*` static libs build warning-free.
  §2.1 confirmed empirically: the four-pair scratch TU including both
  `metis.h` and `precision.hpp` compiles, links and runs, and the mismatched
  combinations fail exactly as predicted —
  `error: conflicting declaration 'using real_t = double'` /
  `note: previous declaration as 'typedef float real_t'` for a wrong
  `REALTYPEWIDTH`, and the `idx_t`/`index_t` `static_assert` for a wrong
  `IDXTYPEWIDTH`. The four-variant build is load-bearing, not defensive.
- **Batch 1 — base. DONE.** `src/base/owned_array.hpp`
  (`OwnedArray<T, NDIM>`, `make_dims`, `adopt_malloc`). Verified through a
  scratch nanobind module across all four precision pairs: buffers survive the
  owning `OwnedArray` going out of scope; rank 1/2/3, `index_t`/`real_t`/`int8`
  dtypes, C-contiguity and row-major layout all correct; `zero_init` zeroes;
  zero-length arrays work; `view()` agrees with `operator()`; `adopt_malloc`
  round-trips a `malloc`'d buffer; double-`release()` and shape overflow raise
  instead of corrupting. No leak over 3000 iterations (maxRSS delta 0 KB), and
  ASAN reports no alloc/dealloc mismatch, use-after-free or double-free —
  its residual leak report is CPython baseline noise (571,893 B for
  `import numpy` alone vs 572,925 B after the full driver, no frames in our
  code).
- **Batch 2 — leaf compute. DONE**, with two scope changes requested during
  implementation (see §11).
  - `src/base/print_debug.hpp` — c_api's `print_instant` + `time_it` became a
    project-wide facility (`print_debug`, `print_debug_time_start`,
    `print_debug_time`, `MANAPY_PRINT_DEBUG*` macros), header-only so any
    module can trace without a CMake change.
  - `compute_cell_center_volume.cpp` went to **src/domain**, not
    src/partitioning: declared in `domain_compute.hpp`, implemented in
    `cpu/compute_cell_center_volume_cpu.cpp` (comments preserved verbatim),
    the two public entry points bound in
    `bindings/compute_cell_center_volume.cpp` and exported from all four
    `python/manapy_compute_*/domain/__init__.py`. The two `halo_*` helpers are
    declared but unbound — src/partitioning will compile the same TU for them
    in Batch 4, exactly as c_api used them internally.
  - `src/partitioning/includes/manapy_part.hpp` + `src/partitioning/src/utils.cpp`
    ported (`binary_search`, `intersect_arr`, `get_max_info`). No CMake target
    yet — there is nothing to bind until Batch 5 — so they are compile-verified
    out of tree for all four pairs instead.
  - Verified: bitwise-equal output vs the c_api reference wheel
    (`manapy_part-1.0.2-cp314`) on random 4000-cell meshes covering triangle,
    quad, tetrahedron, pyramid and hexahedron, for all four precision pairs,
    both 2D and 3D, plus degenerate zero-vertex rows.
- **Batch 3 — struct. DONE.** `includes/local_domain_struct.hpp` +
  `src/local_domain_struct.cpp`. The 18 tables went from raw
  `PyArray<T, Dim> *` to `OwnedArray<T, N>` held by value (§4.3 explains why
  neither `optional` nor `unique_ptr`), `tuple_res` from
  `PyObject *` to `nb::object`, and `create_tuple()` from `Py_BuildValue` to
  `nb::make_tuple`; the member documentation is the original's, verbatim. The
  hand-rolled destructor is gone — `optional` and `nb::object` clean up
  themselves — and `free_tables()` is now just `.reset()` on each table.
  `take()` reports a missing table by name instead of segfaulting on a null
  pointer.
  Verified on all four precision pairs (109 checks each): the result is a
  27-entry tuple whose 18 arrays carry the expected dtype, shape, C-contiguity
  and contents in the c_api's exact order (each table filled with its own
  position, and every table given a distinct shape, so a transposed pair would
  show up); the 9 scalars come back with the right values; `max_halo_cell_nodeid`
  correctly stays out of the tuple; arrays outlive their `LocalDomainStruct`;
  a second `create_tuple()` and each missing table raise; no leak over 4000
  build/drop cycles (maxRSS delta 0 KB); ASAN clean for alloc/dealloc mismatch,
  use-after-free and double-free.
  **`Py_BuildValue`'s `"i"` truncation is confirmed fixed**: `max_cell_halophyid`
  round-trips 3_000_000_000 on int64 builds, which would have wrapped before.
- **Batch 4 — partitioning. DONE.** `src/partitioning.cpp` (1256 lines) plus
  `includes/partitioning.hpp` for the `create_sub_domains` declaration — in the
  c_api that lived in `manapy_part.h`, which included `LocalDomainStruct.h`;
  those two headers are split here, so the declaration needing both gets its
  own. `get_result_as_py_list` returns `nb::list`; `VecMapNodes`, the five
  pipeline stages and `handle_periodic_faces` are transcribed with the comments
  unchanged.
  `ArrayView` gained `last()` (1D only) for this: connectivity rows store their
  count in the final slot, `partitioning.cpp` reads and writes it ~25 times, and
  spelling `x(x.size(0) - 1)` at every site is 25 chances to mistype an index.
  Compiles warning-free under `-Wall -Wextra` on all four pairs; the CUDA
  targets still build, since `last()` is `MANAPY_COMPUTE_HOST_DEVICE` like the
  rest of `ArrayView`.
  Verified against the c_api reference wheel: **1931 checks per precision pair,
  all four pairs, zero failures** — every one of the 27 tuple entries (dtype,
  shape, values) for every partition, over 10 cases spanning triangles, quads
  and hexahedra, 2/3/4/5 partitions, and periodic tags in 2D and 3D. The
  periodic path was confirmed to actually run rather than agreeing on a no-op:
  the periodic mesh injects 36 extra exterior halo cells, all of which share no
  node with the partition receiving them, and their centroids are translated by
  ±L (partition 0 at hi-x carries halos at x = 10.33, partition 2 at lo-x at
  x = -0.33) — bit-identical to the reference.
  One addition, flagged in the code: `create_sub_domains` now rejects a `nodes`
  array whose width is not 3. The per-partition `nodes` table is allocated 3
  wide and filled with `nodes.size(1)` columns, and `handle_periodic_faces`
  reads column 2 unconditionally, so any other width read or wrote out of
  bounds.
- **Batch 5 — METIS bindings. DONE.** `bindings/registry.hpp`,
  `bindings/module.cpp` (`NB_MODULE(_core)`),
  `bindings/metis_partitioning.cpp` (`dense_to_csr` + the three `make_n_part_*`
  and their bindings, kept together as they were in the c_api's
  py_manapy_part.cpp) and `bindings/create_local_domains.cpp`. Docstrings are
  the originals, except `create_local_domains`' — rewritten to the real 27-item
  tuple (§5 item 2).
  The METIS output vector is an `OwnedArray`, so it is freed automatically if
  METIS reports an error and handed to NumPy without a copy otherwise; the CSR
  scratch is `std::vector`. That removes every `free()` on every error path,
  along with the eight `Py_XDECREF`s and the `try/catch` that
  `py_create_local_domains` needed. `adopt_malloc` in `owned_array.hpp` is now
  unused — METIS never allocates for us, it fills a caller-provided buffer — so
  it should be deleted rather than kept as untested surface.
- **Batch 6 — packaging. DONE.** The `_partitioningcore_<F>_<I>` CMake targets
  (CPU-only, each linking `metis_i${IBITS}_f${FBITS}`, and compiling
  `src/domain/cpu/compute_cell_center_volume_cpu.cpp` for create_halos' halo_*
  calls), the four `partitioning/__init__.py`, the `_SUBMODULES` entries and
  package docstrings, and the README.
  Verified: all four modules build; lazy `from manapy_compute_64_32 import
  partitioning` works and `create_local_domains` returns 27-item tuples with
  the right dtypes; the generated install rules place `_core.so` at
  `<pkg>/partitioning/`, the same shape as `domain`'s. **§9 check 4 passes** —
  `manapy_compute_32_32.partitioning` and `manapy_compute_64_64.partitioning`
  loaded in one interpreter each return their own index width, so the two METIS
  builds do not cross-bind.
  Not run: a full `pip wheel .`, which rebuilds every CUDA target. The install
  rules and `wheel.packages` roots were checked directly instead.
- **Batch 7 — parity. DONE.** Real manapy meshes read with `Mesh` and tables
  built with `DomainCompute`, so both implementations get exactly what
  production feeds them: `triangles`, `carre` (2744 cells), `hybrid2d`,
  `smallHybrid2D`, `smallTetrahedrons`, `smallHybrid3d` (types 3/4/5 in one
  mesh), `smallCuboid`, at 2/3/4 partitions. **4263 checks per precision pair,
  all four pairs, zero failures.** One `part_vert` feeds both sides, isolating
  the pipeline from §11.4's METIS difference.
  Two things this turned up that the synthetic meshes could not — both
  pre-existing c_api behaviour, see §11.5 and §11.6.

---

## 9. Verification

Parity against the reference implementation, using the prebuilt
`c_api/wheelhouse/manapy_part-1.0.2-*.whl` (or a source build of c_api) in a
separate venv:

1. Take a real mesh (`main.py`'s `rectangles.msh` via `get_mesh`) and, for each
   of the 4 precision pairs:
   - `make_n_part_mesh_dual` / `_nodal` / `make_n_part_graph_k_way` →
     assert **bitwise-equal** partition vectors (METIS is deterministic for a
     fixed seed and identical CSR input, and `dense_to_csr` is unchanged).
   - `create_local_domains` → compare all 27 tuple entries per partition with
     `np.array_equal` (ints) / `np.allclose` (floats: the arithmetic is
     unchanged, so exact equality should hold — investigate any diff rather
     than loosening the tolerance).
   - `compute_cell_center_area_2d` / `_volume_3d` → exact equality.
2. dtype/shape assertions on every returned array (the `int` truncation fix in
   §4.3 means the 9 `max_*` scalars are now Python ints of full width).
3. Leak check: call `create_local_domains` in a loop, watch RSS — the capsule
   deleters are the one genuinely new ownership mechanism.
4. Cross-import check: `import manapy_compute_32_32.partitioning` and
   `manapy_compute_64_64.partitioning` in the same interpreter and run both —
   confirms the METIS symbol isolation of §3.
5. `main.py`-level smoke test through `manapy.domain.Domain.create_domain`.

---

## 10. Risks / open points

| risk | mitigation |
|---|---|
| `real_t` typedef clash with `metis.h` | 4 METIS variants (§2.1) — validated in Batch 0 before anything else is written |
| nanobind implicit dtype conversion differs from `PyArray_FROM_OTF` | spike in Batch 1; fall back to explicit `np.ascontiguousarray` in the Python wrapper |
| METIS/GKlib bloat the repo (55 MB) | prune non-compiled trees; record upstream commit in `third_party/README.md` |
| GKlib symbol collisions across the 4 loaded `.so`s | `C_VISIBILITY_PRESET hidden` + Python's `RTLD_LOCAL`; verified by check 4 in §9 |
| Structural deviation from the repo's `bindings/cpu/common` convention | explicit, per the instruction to keep c_api's `src` layout; documented at the top of `src/partitioning/README.md` |
| Return-by-allocation deviates from the in-place convention | unavoidable (output sizes unknown to the caller); documented in §4.2 |
| Stale 22-item docstring propagating | §5 item 2 — write the 27-item list from `create_tuple`, not from the old docs |
| Wrong-dtype **output** arrays are silently discarded project-wide (§11) | fixed for the Batch 2 bindings with `.noconvert()`; 57 pre-existing binding files still affected |
| cibuildwheel time grows (4 more targets, each with a full METIS build) | METIS objects are per-variant, so 4 builds; acceptable, but if CI time bites, `metis_i32_f32` and `metis_i64_f32` etc. cannot be shared — only the `REALTYPEWIDTH` split is avoidable, and only by the rejected C-shim approach |

---

## 11. Scope changes made during Batch 2

Two deviations from §1/§4, requested while Batch 2 was being implemented, plus
one problem found by its parity test.

### 11.1 `print_debug` is project-wide

c_api's `print_instant` (formatted line) and `time_it` (acc/delta timing) were
two functions inside its `utils.cpp`. They are now one facility in
`src/base/print_debug.hpp`, available to every module rather than only to the
partitioner:

| c_api | now |
|---|---|
| `print_instant("...")` | `print_debug("...")` |
| `time_it("")` | `print_debug_time_start()` |
| `time_it("phase")` | `print_debug_time("phase")` |
| `DEBUG_PRINT_INSTANT(...)` | `MANAPY_PRINT_DEBUG(...)` |
| `DEBUG_TIME_IT(msg)` | `MANAPY_PRINT_DEBUG_TIME(msg)` |
| `manapy_debug_timing_enabled()` | `print_debug_enabled()` |

Header-only, so including it is the whole integration — no CMake change, no
library. Still gated on the environment (`MANAPY_DEBUG` preferred; the legacy
`MANAPY_DEBUG_TIMING` / `MANAPY_TIMING_DEBUG` names still work) and still
routed through Python's `sys.stdout` so tracing interleaves with the caller's
own output; it falls back to C stdout if Python is unavailable. Requires the
GIL, which every kernel call site already holds. `MANAPY_DISABLE_PRINT_DEBUG`
compiles the macros away entirely.

Two behaviours kept deliberately: the hardcoded `"C\t[Rank 0]: "` prefix (this
layer has no MPI rank; the env var accepting `all`/`rank0` anticipated
filtering that was never implemented), and the timer's acc/delta semantics. The
one change is `steady_clock` instead of `gettimeofday`, so a clock adjustment
mid-run cannot produce a negative delta.

### 11.2 The geometry kernels live in `src/domain`

`compute_cell_center_volume.cpp` is mesh geometry, so it went to `src/domain`
rather than `src/partitioning`. Consequences:

- Python surface is `manapy_compute_<F>_<I>.domain.compute_cell_center_area_2d`
  / `.compute_cell_center_volume_3d`, not `.partitioning.*`. §5's binding list
  shrinks to the four METIS entry points.
- `compute_halo_cell_center_area_2d` / `_volume_3d` stay unbound but are
  declared in `domain_compute.hpp`; the Batch 5/6 partitioning target must add
  `src/domain/cpu/compute_cell_center_volume_cpu.cpp` to its source list (and
  `src/domain` to its include dirs) so `create_halos` can call them. The same
  TU compiled into two targets is fine — it is CPU-only and precision-macro
  driven like everything else.
- Element-geometry changes now have one home. The four functions carry the
  original "** This code is the same as ... **" cross-references, which remain
  accurate.

One addition to otherwise-verbatim bodies: each loop now rejects a vertex count
larger than the fixed `p[4]`/`p[8]` stack buffer. The original copies
`nb_vertex` vertices straight from mesh data with no check, so a malformed row
smashed the stack — reachable from Python in c_api too. Valid meshes are
unaffected; malformed ones get a `ValueError`.

### 11.3 Found by the parity test: outputs need `.noconvert()`

§4.1 flagged nanobind's implicit dtype conversion as a "verify early" item. It
is real, and worse on the way out than on the way in: passing an **output**
array of the wrong dtype produces **no error and no results**. nanobind casts
it to a temporary, the kernel dutifully fills the temporary, and the caller's
array comes back untouched.

Confirmed on a pre-existing binding — `count_max_node_cellid` on an int64 build
returns `[5 5 5]` into an int64 `res` and `[0 0 0]` into an int32 one, silently:

```python
m.count_max_node_cellid(cells, np.zeros(3, dtype=np.int64))  # -> [5 5 5]
m.count_max_node_cellid(cells, np.zeros(3, dtype=np.int32))  # -> [0 0 0]
```

The fix is `nb::arg("...").noconvert()` on mutable arguments only; read-only
inputs should keep converting, which is what `PyArray_FROM_OTF` gave the c_api.
Applied to the Batch 2 bindings and written into
[../domain/README.md](../domain/README.md)'s recipe.

**Resolved project-wide.** `.noconvert()` is now on **187 mutable arguments
across 56 binding files** in `core`, `domain`, `boundary`, `solvers` and
`partitioning` — every bound function that writes into a caller-supplied array,
CPU and CUDA alike. Read-only inputs deliberately keep implicit conversion.

The mapping was done mechanically rather than by eye: for each `m.def`, the
bound function's parameter list is parsed and each `nb::arg` is matched
positionally against its declared type, so `.noconvert()` lands only on the
mutable aliases (`FVec`, `FMat`, `FTensor`, `IVec`, `IMat`, `I8Vec`, `DFVec`)
and never on a `C*`/`DC*` one. Scalars are skipped. Verified by re-running the
same pass afterwards: it finds nothing left to change.

All 32 modules rebuild warning-free, and every suite still passes. The
behaviour is confirmed directly on the function that first exposed it:

```python
m.count_max_node_cellid(cells, np.zeros(3, dtype=np.int64))  # -> [5 5 5]
m.count_max_node_cellid(cells, np.zeros(3, dtype=np.int32))  # -> TypeError
m.count_max_node_cellid(cells.astype(np.int32), out64)       # -> [5 5 5], input still converts
```

### 11.4 METIS's partition output can differ from the c_api's on float64 packages

Found by Batch 5's parity test. `make_n_part_graph_k_way` on the float64/int64
package returned a partition differing from the c_api's on 6 of 160 vertices.
It is not a porting bug — it follows directly from §2.1's four-variant METIS
build:

| | METIS `IDXTYPEWIDTH` | METIS `REALTYPEWIDTH` |
|---|---|---|
| c_api | 32 or 64 | **always 32** |
| here | follows `MANAPY_COMPUTE_INT_BITS` | follows `MANAPY_COMPUTE_FLOAT_BITS` |

So the float32 packages link a METIS identical to the c_api's and reproduce it
bit-for-bit, while the float64 packages link a `REALTYPEWIDTH=64` METIS whose
internal floating-point tie-breaks during refinement can land on a different
partition. Confirmed by isolating the variable — same index width, same input
graph, only `REALTYPEWIDTH` changed:

```
ours(REAL=32, IDX=64) == c_api : True
ours(REAL=64, IDX=64) == c_api : False   (6/160 vertices differ)
  c_api    sizes=[54, 53, 53] edgecut=16
  REAL=64  sizes=[54, 53, 53] edgecut=15
```

Both partitions are valid and equally balanced; the REAL=64 one happens to cut
one fewer edge. Nothing downstream breaks — `create_local_domains` stays
bit-exact against the reference when both are fed the same `part_vert`, so the
ported pipeline itself is unaffected.

It is still worth a decision, because the conceptual story is off: partitioning
is integer work over connectivity, and METIS's `real_t` (used only for
`tpwgts`/`ubvec`, which are always `nullptr` here) has nothing to do with the
precision of the mesh coordinates. Tying the two means the same mesh can
decompose differently in a float32 and a float64 build, and float64 users no
longer reproduce the old partitions.

Two ways out, if that matters:

1. **Keep it.** Four METIS builds, as requested. Accept that float64 packages
   produce different-but-valid partitions from the c_api's, and note it in the
   release notes so nobody chases a phantom regression.
2. **Pin `REALTYPEWIDTH=32` everywhere** and dodge the `real_t` collision with a
   shim instead: one small TU includes `metis.h` and nothing else, exposing
   wrappers typed with explicit `int32_t`/`int64_t`, so no translation unit
   ever sees both `metis.h` and `precision.hpp`. Two METIS builds instead of
   four, exact c_api parity on every pair, at the cost of one indirection
   layer. This is the alternative §2.1 rejected — it looks better now that the
   cost of the four-way split is visible.

### 11.5 Ragged tables were handed to Python with uninitialized padding

`cells`, `phy_faces` and `halo_halosext` are ragged: a row holds `count`
entries plus the count itself in the last slot, and everything between is never
written. The c_api allocated them with `PyArray_SimpleNew` and so handed those
padding slots to Python as whatever was on the heap.

Invisible on a uniform mesh (every row fills its width), obvious on a hybrid
one. A tetra row from `smallHybrid3d.msh`, ours vs the c_api:

```
ours: [ 6  2 10  8 |  1  8 12  2 | 4]     <- 4 nodes, padding, count
ref : [ 6  2 10  8 | 11  1  7  9 | 4]
```

Audited across every table and partition: **0 differences in defined slots,
all differences in padding**. So it was never a correctness problem for
consumers, which read only `count` entries — but it made output
nondeterministic run to run and leaked heap contents into NumPy arrays.

The three allocations now pass `zero_init = true`. Nothing can legitimately
read those slots, so no defined value changes; the cost is a memset that is
noise next to the pipeline. Batch 7 asserts both halves: defined slots equal
the reference, and our padding is zero.

### 11.6 The c_api's int64 `create_local_domains` reads an uninitialized `dim`

`py_create_local_domains` declares

```c
idx_t nb_parts = 0;
idx_t dim;                       // uninitialized
PyArg_ParseTuple(args, "OOOOOOOOii", ..., &nb_parts, &dim);
```

`"i"` writes a C `int` — 4 bytes — into an 8-byte `idx_t` on the int64 builds,
so `dim`'s upper half is stack garbage. `nb_parts` survives only because it is
initialized to 0. Depending on what the caller left on the stack, the call
either works or raises `ValueError: dim must be 2 or 3`: it does the former
from a shallow test harness and the latter under manapy's deeper call stack,
which is how Batch 7 hit it.

This is the input-side twin of the `"i"` output truncation noted in §4.3, and
it means `manapy_part64_32` / `manapy_part64_64` are not reliably usable as a
reference — Batch 7 compares int64 builds against the c_api's *int32* module
instead (same mesh, same algorithm, values identical, only the index width
differs). The port is unaffected: nanobind casts `index_t` at its real width,
so nothing here is uninitialized.
