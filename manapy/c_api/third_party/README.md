# Vendored third-party sources

Both trees are unmodified upstream sources, imported from
`manapy/c_api/third_party` (which vendored them in turn). Only the parts that
are actually compiled were kept — see "Pruned" below.

| library | version | upstream |
|---|---|---|
| METIS | 5.2.1 | https://github.com/KarypisLab/METIS |
| GKlib | (bundled with METIS 5.2.1; no version stamp of its own) | https://github.com/KarypisLab/GKlib |

Both are Apache-2.0 licensed; `METIS/LICENSE` and `GKlib/LICENSE.txt` are kept
verbatim.

## What is built

`CMakeLists.txt` here builds four static libraries — `metis_i32_f32`,
`metis_i32_f64`, `metis_i64_f32`, `metis_i64_f64` — one per
(`IDXTYPEWIDTH`, `REALTYPEWIDTH`) pair. The header comment in that file
explains why the float width has to be parameterised too (METIS declares a
global `real_t` typedef that would otherwise collide with the one in
`src/base/precision.hpp`).

Sources compiled: `GKlib/src/*.c` and `METIS/libmetis/*.c`.

## Pruned

Not vendored, because nothing in this project compiles or runs them:

- `METIS/`: `graphs/` (13 MB of sample meshes), `manual/`, `programs/` (the
  `gpmetis`/`ndmetis` CLIs), `utils/`, `conf/`, top-level `CMakeLists.txt`,
  `Makefile`, `vsgen.bat`, `BUILD-Windows.txt`
- `GKlib/`: `apps/`, `scripts/`, `cmake/`, `conf/`, top-level `CMakeLists.txt`,
  `Makefile`, `GKlibConfig.cmake.in`

The `CMakeLists.txt` files inside `METIS/include/` and `METIS/libmetis/` are
upstream's own and are left in place untouched; this project does not
`add_subdirectory` them.

## Updating

Re-copy the two directories from a fresh upstream checkout, re-apply the
pruning above, and update the version table. There are no local patches to
carry forward.
