# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

METIS is a C library (C99) for partitioning graphs/meshes and computing fill-reducing orderings of sparse matrices. It implements the multilevel paradigm: coarsen → initial partition → refine/uncoarsen. Version 5.2.1, authored by George Karypis. Licensed under Apache 2.0.

**External dependency:** [GKlib](https://github.com/KarypisLab/GKlib) must be installed before building METIS.

## Build Commands

```bash
# Configure (creates build/ directory, runs CMake)
make config prefix=~/local gklib_path=~/local

# Build and install
make install

# Common config variants
make config shared=1                    # shared library instead of static
make config i64=1 r64=1                 # 64-bit indices and floats
make config debug=1 assert=1 gdb=1     # debug build with assertions

# Clean
make clean       # remove object files
make distclean   # remove entire build/ directory
```

The top-level Makefile is a proxy that translates options into CMake flags and runs CMake inside `build/`.

## Testing

There is no integrated test suite. Test manually using the CLI programs against sample graphs in `graphs/`:

```bash
./build/programs/gpmetis graphs/4elt.graph 4      # partition graph into 4 parts
./build/programs/ndmetis graphs/4elt.graph         # nested dissection ordering
./build/programs/mpmetis graphs/metis.mesh 4       # mesh partitioning
./build/programs/graphchk graphs/4elt.graph        # validate graph file
```

Note: `test/mtest.c` exists but uses the old METIS 4.x API and has a non-functional build system.

## Architecture

### Directory Structure

- `include/metis.h` — Public API: types (`idx_t`, `real_t`), enums, function prototypes
- `libmetis/` — Core library (~45 source files, one algorithm per file)
- `programs/` — Six CLI tools (`gpmetis`, `ndmetis`, `mpmetis`, `m2gmetis`, `graphchk`, `cmpfillin`)
- `graphs/` — Sample graph/mesh files for testing
- `conf/gkbuild.cmake` — Shared CMake config (compiler flags, platform detection)

### Include Chain

`metislib.h` is the single master internal include — every `libmetis/*.c` file includes only this. It pulls in:
`GKlib.h` → `metis.h` → `rename.h` → `gklib_defs.h` → `defs.h` → `struct.h` → `macros.h` → `proto.h`

Programs use `metisbin.h` which includes both the internal libmetis headers and program-specific `defs.h`, `struct.h`, `proto.h`.

### Core Data Structures (`libmetis/struct.h`)

- **`graph_t`** — CSR graph (xadj/adjncy arrays), plus partition state (where, pwgts, boundary lists, refinement info). Coarsening levels form a linked list via `coarser`/`finer` pointers.
- **`ctrl_t`** — Algorithm control: operation type, coarsening/refinement methods, balance tolerances, workspace memory core, neighbor pools.
- **`mesh_t`** — Element-node connectivity in CSR format (eptr/eind).

### Key Algorithmic Files

| File | Algorithm |
|------|-----------|
| `kmetis.c` | Direct k-way partitioning (METIS_PartGraphKway) |
| `pmetis.c` | Recursive bisection (METIS_PartGraphRecursive) |
| `ometis.c` | Nested dissection ordering (METIS_NodeND) |
| `coarsen.c` | Graph coarsening (RM, SHEM, 2-hop matching) |
| `initpart.c` | Initial partitioning (grow bisection, random) |
| `fm.c` | FM 2-way refinement |
| `kwayfm.c` | Greedy k-way refinement (cut and volume) |
| `sfm.c` | Separator FM refinement |
| `contig.c` | Contiguity enforcement |
| `minconn.c` | Subdomain connectivity minimization |
| `mesh.c` | Mesh-to-graph conversion |

### Public API

All functions return `METIS_OK`, `METIS_ERROR_INPUT`, `METIS_ERROR_MEMORY`, or `METIS_ERROR`. Options are passed as `idx_t options[METIS_NOPTIONS]` initialized to -1 via `METIS_SetDefaultOptions()`.

Key entry points: `METIS_PartGraphRecursive`, `METIS_PartGraphKway`, `METIS_PartMeshDual`, `METIS_PartMeshNodal`, `METIS_NodeND`, `METIS_MeshToDual`, `METIS_MeshToNodal`.

## Coding Conventions

- **Naming:** Public API: `METIS_PascalCase`. Internal functions: `PascalCase`. Types: `lowercase_t` (enums: `_et`). Constants/macros: `SCREAMING_SNAKE_CASE`.
- **Configurable widths:** `idx_t` is int32 or int64, `real_t` is float or double, set at build time via `IDXTYPEWIDTH`/`REALTYPEWIDTH`.
- **Symbol isolation:** All internal symbols prefixed `libmetis__` via `rename.h` and `gklib_rename.h` to avoid collisions (critical for ParMETIS).
- **Memory:** GKlib-based (`gk_malloc`/`gk_free`). Stack workspace via `WCOREPUSH`/`WCOREPOP`. Free flags on graph arrays track ownership. `gk_free()` uses `LTERM` sentinel.
- **Error handling:** Public API uses `gk_sigtrap()`/`gk_sigcatch()` with `goto SIGTHROW` pattern. Internal errors use `gk_errexit()`.
- **Patterns:** `IFSET(ctrl->dbglvl, ...)` for conditional debug output. `GETOPTION(options, key, default)` for safe option extraction. `BNDInsert`/`BNDDelete` for boundary management.
- **Fortran bindings:** `frename.c` provides four naming variants per API function (ALLCAPS, alllower, alllower_, alllower__).
