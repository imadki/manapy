# manapy_compute

`manapy_compute` is the native compute layer used by Manapy. It contains
finite-volume and unstructured-mesh kernels implemented in C++20 and CUDA,
exposed to Python with [nanobind](https://github.com/wjakob/nanobind).

The project builds four Python packages, one for every floating-point/index
precision pair:

| Package | Floating-point data | Mesh indices |
| --- | --- | --- |
| `manapy_compute_32_32` | `float32` | `int32` |
| `manapy_compute_32_64` | `float32` | `int64` |
| `manapy_compute_64_32` | `float64` | `int32` |
| `manapy_compute_64_64` | `float64` | `int64` |

The suffix is always `<float bits>_<int bits>`. For example,
`manapy_compute_32_64` uses `float32` values and `int64` indices.

CPU entry points accept NumPy arrays. CUDA entry points have a `_cuda` suffix
and accept CuPy arrays without copying their device buffers, through
nanobind's DLPack-compatible ndarray interface. Submodules are separate shared
libraries and are imported lazily, so importing the top-level package does not
load every extension or the CUDA runtime.

## Contents

| Python module | Purpose | Backend |
| --- | --- | --- |
| `core` | 2D/3D gradients, interpolation, limiters, face-to-cell and cell-to-face kernels | CPU + CUDA |
| `domain` | Mesh connectivity, geometry, ghost/halo tables and periodic information | CPU; optional OpenMP for selected loops |
| `boundary` | Scalar and free-slip ghost/halo-ghost boundary conditions | CPU + CUDA |
| `partitioning` | Graph/mesh partitioning and local-domain construction | CPU, using METIS/GKlib |
| `solvers.advec` | Explicit advection kernels | CPU + CUDA |
| `solvers.advecdiff` | Explicit advection-diffusion kernels | CPU + CUDA |
| `solvers.diffusion` | Explicit diffusion kernels | CPU + CUDA |
| `solvers.utils` | Shared initialisation and update kernels | CPU, with CUDA for supported functions |

Example imports:

```python
from manapy_compute_64_64 import core, domain, partitioning
from manapy_compute_64_64.solvers import advecdiff

# CPU functions use NumPy arrays.
core.cell_gradient_2d(...)

# GPU functions use CuPy arrays and have a `_cuda` suffix.
core.cell_gradient_2d_cuda(...)
```

Function signatures and docstrings are available through Python:

```bash
python -c "from manapy_compute_64_64 import core; help(core.cell_gradient_2d)"
```

## Technology and dependency model

- **C++20 and CUDA:** CPU and GPU implementations share small
  `MANAPY_COMPUTE_HOST_DEVICE` routines where appropriate. GPU kernels use the
  CUDA runtime API.
- **nanobind:** builds the CPython extension modules and provides typed NumPy
  and device-array bindings.
- **NumPy:** required at runtime (`numpy>=1.17`) and used by CPU entry points.
- **CuPy:** optional runtime dependency for CUDA entry points. The declared
  extra installs `cupy-cuda12x`.
- **METIS 5.2.1 and GKlib:** vendored under `third_party/`, compiled as static
  libraries, and linked into `partitioning`. No system METIS or GKlib install
  is required. Four variants are built so both METIS integer width and its
  `real_t` width match each package's precision.
- **scikit-build-core and CMake:** `pyproject.toml` is the Python build entry
  point; CMake owns native compilation.
- **OpenMP:** optional. When found, it accelerates selected `domain` loops;
  the build remains correct without it.

The project is Linux-only. `pyproject.toml` currently declares CPython 3.8 or
newer and the release-wheel configuration targets glibc-based Linux on
`x86_64`. PyPy, Windows, macOS and musllinux/Alpine wheels are not configured.

## Prerequisites for a source build

Install these system tools before building locally:

| Tool | Requirement |
| --- | --- |
| Python | CPython 3.8+ with development headers and `venv`/`pip` |
| CMake | 3.26+ |
| C/C++ compiler | C compiler plus a C++20-capable compiler supported by the selected CUDA toolkit |
| CUDA toolkit | `nvcc`, CUDA headers and CUDA runtime development libraries |
| Build runner | Ninja is recommended; Make also works |

CUDA 12.8 or newer is recommended because it builds the complete architecture
set, including Blackwell. With a toolkit older than 12.8, CMake omits
`sm_100`/`sm_120` and uses a `compute_90` PTX fallback. The current fixed list
also contains `sm_89` and `sm_90`, so the toolkit must understand those
targets. In practice, use the CUDA 12.x toolchain used by the manylinux build
unless you intentionally maintain another supported toolchain.

The CMake project discovers CUDA before enabling the CUDA language. If several
toolkits are installed, select one explicitly to prevent `nvcc` and `ptxas`
from different installations being mixed:

```bash
cmake -S . -B build/manual -G Ninja \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_BUILD_TYPE=Release
```

Even a CPU-only extension target currently needs a discoverable CUDA toolkit,
because CUDA is enabled for the project as a whole.

## Build and install from source

The simplest isolated development setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

This invokes the PEP 517 backend, installs its Python-side build dependencies
in an isolated environment, builds every precision/module extension, creates
a wheel, and installs it. CMake, a compiler and the CUDA toolkit remain system
requirements.

For an editable checkout, optionally including CuPy:

```bash
python -m pip install -e '.[gpu]'
```

For repeated native builds, install the build dependencies in the active
environment and disable build isolation:

```bash
python -m pip install 'scikit-build-core>=0.10' 'nanobind>=2.1' ninja
python -m pip install -e . --no-build-isolation
```

To select a non-default CUDA installation during a Python build:

```bash
CUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  python -m pip install .
```

Verify the installed package with a lightweight import check:

```bash
python -c "from manapy_compute_64_64 import core, domain, partitioning; print('import ok')"
```

## Build a wheel now and install it later

This is the cleanest way to separate compilation from installation.

```bash
python -m pip install --upgrade build
python -m build --wheel
```

The wheel is written to `dist/`. Installing that already-built wheel with
Python does not compile the project again:

```bash
python -m pip install --force-reinstall dist/manapy_compute-0.1.0-*.whl
```

Use the filename actually produced in `dist/` after changing the project
version. Add `--no-deps` only when NumPy is already managed separately.

## Build only the `.so` extension libraries

Use CMake directly when you want native shared objects for development or
testing but do not want to create or install a Python wheel.

First install nanobind in the Python environment CMake will use, then
configure the native build:

```bash
python -m pip install 'nanobind>=2.1' ninja
cmake -S . -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
```

Build every shared library:

```bash
cmake --build build/manual --parallel
find build/manual -type f -name '_core*.so'
```

Or build only one extension. Target names end in `<float bits>_<int bits>`:

| Python submodule | CMake target prefix | Example for `float64`/`int64` |
| --- | --- | --- |
| `core` | `_gradcore_` | `_gradcore_64_64` |
| `domain` | `_domaincore_` | `_domaincore_64_64` |
| `partitioning` | `_partitioningcore_` | `_partitioningcore_64_64` |
| `boundary` | `_boundarycore_` | `_boundarycore_64_64` |
| `solvers.advec` | `_adveccore_` | `_adveccore_64_64` |
| `solvers.advecdiff` | `_advecdiffcore_` | `_advecdiffcore_64_64` |
| `solvers.diffusion` | `_diffusioncore_` | `_diffusioncore_64_64` |
| `solvers.utils` | `_solversutils_` | `_solversutils_64_64` |

```bash
cmake --build build/manual --parallel --target _gradcore_64_64
```

The result is placed below
`build/manual/manapy_compute_64_64/<submodule>/`. A raw `.so` is not a complete
installation: the package also needs its Python `__init__.py` files and the
correct directory layout. Build a wheel and install it as shown above when the
goal is a normal Python installation.

## Supported NVIDIA GPU architectures

GPU-capable extensions are compiled for these architectures:

| CMake architecture | Generated code | NVIDIA generation/examples |
| --- | --- | --- |
| `75-real` | `sm_75` SASS | Turing: RTX 20 series, T4 |
| `80-real` | `sm_80` SASS | Ampere: A100 |
| `86-real` | `sm_86` SASS | Ampere: RTX 30 series, A40 |
| `89-real` | `sm_89` SASS | Ada: RTX 40 series, L40 |
| `90-real` | `sm_90` SASS | Hopper: H100 |
| `100-real` | `sm_100` SASS | Blackwell: B100/B200; CUDA 12.8+ |
| `120-real` | `sm_120` SASS | Blackwell: RTX 50 series; CUDA 12.8+ |
| `120-virtual` | `compute_120` PTX | Forward-compatible fallback; CUDA 12.8+ |

On CUDA toolkits older than 12.8, the last three entries are replaced by a
`compute_90` PTX fallback. Runtime CUDA calls still require a compatible
NVIDIA driver and GPU. CPU functions can be used without a GPU.

## manylinux wheels

Release wheels are built with `cibuildwheel` using the settings in
`pyproject.toml`:

- Linux `x86_64` only;
- CPython 3.8+ (`cp3*`, filtered by the project's Python requirement);
- `manylinux_2_28` based on AlmaLinux 8;
- musllinux skipped;
- CUDA 12.8 `nvcc` and CUDA runtime development files installed inside the
  container;
- import-level smoke testing, which does not require a GPU.

Docker must be installed and usable by the current user. The host does not
need a CUDA toolkit or GPU for this containerized build:

```bash
python -m pip install --upgrade cibuildwheel
python -m cibuildwheel --platform linux
```

Repaired wheels are written to `wheelhouse/`.

### CUDA library policy in wheels

- `libcudart.so.12` is linked dynamically and bundled into each repaired
  wheel by `auditwheel`. Users of a wheel do not need the CUDA toolkit.
- `libcuda.so` and `libcuda.so.1` are explicitly excluded. They belong to the
  NVIDIA driver installed on the target machine and must never be shipped in
  a Python wheel.
- CUDA-capable extensions can be imported on a GPU-less machine; an actual
  `_cuda` call requires a compatible NVIDIA driver and GPU.

## Publish a release to PyPI

Publishing is a release operation, not part of a normal development build.
Before uploading:

1. Update `project.version` in `pyproject.toml`; PyPI does not allow a version
   filename to be replaced after upload.
2. Verify a clean source checkout and run the relevant CPU/GPU correctness
   checks.
3. Build portable manylinux wheels with `cibuildwheel`, not an unrepaired
   wheel produced directly on the maintainer's workstation.
4. Build the source distribution and validate every artifact.
5. Check the `.toml` for manapy to ensure they include the new version tag.
6. If you make any changes to the `VariableCompute`, `DomainCompute`, or `PartitioningCompute` functions, don't forget to run the dedicated pytest suite.

```bash
python -m pip install --upgrade build cibuildwheel twine
python -m cibuildwheel --platform linux
python -m build --sdist
python -m twine check wheelhouse/*.whl dist/*.tar.gz
```

Test the exact wheels locally in a fresh virtual environment before upload.
Then upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi wheelhouse/*.whl dist/*.tar.gz
```

After checking the TestPyPI release metadata, upload the same immutable files
to production PyPI:

```bash
python -m twine upload wheelhouse/*.whl dist/*.tar.gz
```

Use a PyPI API token (or trusted publishing in CI). Do not store credentials
in this repository or place a token directly in a shell command. Tag the same
version in Git only after the artifacts and metadata have been verified.

## Repository layout

```text
pyproject.toml                         Python metadata and build/release configuration
CMakeLists.txt                         All native extension targets and CUDA architectures
src/base/                              Precision, array-view, CUDA-launch and debug helpers
src/core/                              Core mesh/field kernels (CPU + CUDA)
src/domain/                            Mesh connectivity and geometry kernels (CPU)
src/boundary/                          Boundary-condition kernels (CPU + CUDA)
src/partitioning/                      METIS-backed partitioning (CPU)
src/solvers/                           Advection/diffusion solver modules (CPU + CUDA)
third_party/GKlib/                     Vendored GKlib sources
third_party/METIS/                     Vendored METIS 5.2.1 sources
python/manapy_compute_<F>_<I>/         Python packages for all four precision pairs
```

Within a typical CPU + CUDA module:

```text
common/<kernel>_common.hpp             Shared host/device element or face operation
cpu/<kernel>_cpu.cpp                   CPU loop and public native entry point
gpu/<kernel>_cuda.cu                   CUDA kernel and launcher
bindings/<kernel>.cpp                  nanobind CPU/CUDA wrappers
bindings/registry.hpp                  Registration declarations
bindings/module.cpp                    NB_MODULE(_core) definition
```

`domain` is CPU-only and keeps the implementation in its CPU source.
`partitioning` deliberately preserves its coarser, ported METIS structure.
`solvers` adds another directory level for each solver.

## Adding a new function

Start by deciding:

1. Which public submodule owns the function.
2. Whether it is CPU-only or CPU + CUDA.
3. Whether it is a Python API function or an internal helper.
4. Which precision-dependent types it needs (`real_t`, `index_t`, and the
   existing ndarray aliases in `src/base/manapy_compute_types.hpp`).

Then follow the module-specific guide. These documents are the detailed source
of truth and point to working examples beside the new code:

| Destination | Guide |
| --- | --- |
| `core` | [`src/core/README.md`](src/core/README.md) |
| `domain` | [`src/domain/README.md`](src/domain/README.md) |
| `boundary` | [`src/boundary/README.md`](src/boundary/README.md) |
| any solver or a new solver submodule | [`src/solvers/README.md`](src/solvers/README.md) |
| `partitioning` | [`src/partitioning/Steps.md`](src/partitioning/Steps.md) and [`src/partitioning/README.md`](src/partitioning/README.md) |

For a normal CPU + CUDA function, the complete change usually consists of:

1. Put the per-element/per-face computation in a shared
   `MANAPY_COMPUTE_HOST_DEVICE` header.
2. Add the CPU loop and declaration.
3. Add the CUDA grid-stride kernel, launcher and declaration.
4. Add nanobind CPU and CUDA wrappers. GPU APIs use the `_cuda` suffix and
   device-array aliases.
5. Register the function in `bindings/registry.hpp` and `bindings/module.cpp`.
6. Add every new source file to the correct CMake target.
7. Re-export the public name from all four
   `python/manapy_compute_<F>_<I>/.../__init__.py` packages and keep `__all__`
   sorted.
8. Build all four precision variants of the affected module.
9. Compare CPU and GPU results against an independent Python/NumPy reference
   on inputs that exercise every branch; an import-only test is not enough.
10. update `README.md#Contents` in `c_api`

### Contributor contract for humans and coding agents

Keep these invariants when extending the project:

- Do not duplicate CPU and GPU arithmetic when it can live in one portable
  common routine.
- Treat mutable output arrays specially in nanobind: mark them `.noconvert()`
  so a dtype mismatch cannot create and modify a discarded temporary. Read-only
  inputs may remain convertible when the existing API does so.
- Reuse `ArrayView`, `make_view`, the `CF*`/`F*`/`CI*`/`I*` aliases, and the
  `D*` device aliases before introducing another binding type.
- CUDA wrappers must use the shared launch-error handling in
  `src/base/cuda_launch.hpp`. Do not add unconditional
  `cudaDeviceSynchronize()` calls.
- A helper that is not part of the Python API does not need a binding,
  registry entry or `__init__.py` export.
- Keep package precision order as float bits first, integer bits second. METIS
  uses a different historical ordering, so follow the mapping documented in
  `src/partitioning/Steps.md` when touching partitioning.
- Change source and package files, not generated files below `build/`, `dist/`
  or `wheelhouse/`.
- Update the relevant module README when a new layout rule or non-obvious
  design decision is introduced.

## Debugging

Debug tracing is disabled by default. Enable it with:

```bash
MANAPY_DEBUG=1 python your_program.py
```

`MANAPY_DEBUG_TIMING` and `MANAPY_TIMING_DEBUG` are retained as legacy aliases.
Accepted truthy values include `1`, `true`, `yes`, `on`, `all` and `rank0`.

CUDA launches are asynchronous by default. To synchronize after every launch
and surface an error at the kernel that caused it:

```bash
MANAPY_CUDA_SYNC=1 python your_program.py
```

Use synchronous mode for diagnosis, not production performance. Native debug
messages should go through `src/base/print_debug.hpp` and should not leave
unconditional output in release code.
