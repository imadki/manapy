# manapy_compute

2D unstructured-mesh cell-gradient kernels (least squares) implemented once as a
shared `__host__ __device__` routine and run both in C++ on the CPU and as a CUDA
kernel on the GPU — exposed to Python with [nanobind](https://github.com/wjakob/nanobind).
NumPy arrays feed the CPU path; CuPy arrays feed the CUDA path zero-copy via DLPack.

The kernel is shipped once per precision pair as
`manapy_compute_<float bits>_<int bits>` (e.g. `manapy_compute_32_64` = float32
data, int64 indices), each exposing `cell_gradient_2d` (CPU) and
`cell_gradient_2d_cuda` (GPU).

Built with CMake through [scikit-build-core](https://scikit-build-core.readthedocs.io/)
(`pyproject.toml` is the only build entry point). Linux-only, CPython 3.8+.

## Layout

```
pyproject.toml                       scikit-build-core + cibuildwheel config
CMakeLists.txt                       C++/CUDA build, GPU arch list, cudart linkage
src/base/array_view.hpp              host/device non-owning strided array view
src/base/precision.hpp               real_t / index_t per precision pair
src/base/manapy_compute_types.hpp    nanobind ndarray aliases (CPU + CUDA) + make_view
src/base/owned_array.hpp             owning array that hands its buffer to NumPy
src/base/print_debug.hpp             project-wide debug tracing (env-gated)
src/core/variable_compute.hpp     shared per-cell math + CPU entry point decl
src/core/variable_compute.cpp     CPU loop over cells
src/core/variable_compute.cu      CUDA kernel + launcher (one thread per cell)
src/core/variable_compute_2d_cuda.hpp   GPU launcher declaration
src/core/bindings/cell_gradient_2d.cpp  nanobind module (_core), CPU + GPU bindings
src/domain/                          mesh connectivity/geometry kernels (CPU)
src/boundary/                        boundary-condition kernels (CPU + CUDA)
src/solvers/                         advec / advecdiff / diffusion / utils
src/partitioning/                    METIS-backed domain decomposition (CPU)
third_party/                         vendored METIS + GKlib, built four ways
python/manapy_compute_*/             Python packages wrapping each precision's _core
main.py                              Example driving the CPU path
```

Each `manapy_compute_<float bits>_<int bits>` package exposes its kernels as
submodules — `core`, `domain`, `boundary`, `partitioning` and `solvers` — loaded
lazily, so importing one does not dlopen the others (nor pull in libcudart for
the CPU-only ones).

### Debug tracing

Every module can trace through `src/base/print_debug.hpp`, off unless one of
`MANAPY_DEBUG`, `MANAPY_DEBUG_TIMING` or `MANAPY_TIMING_DEBUG` is set to a
truthy value (`1`, `true`, `yes`, `on`, `all`, `rank0`):

```bash
MANAPY_DEBUG=1 python main.py
```

## Local build

Needs: CUDA toolkit (nvcc), CMake ≥ 3.26, a C++17 compiler.

```bash
pip install .          # or: pip install -e . for development
python main.py
pip install cupy-cuda12x   # for the GPU path (or: pip install .[gpu])
```

## GPU architecture coverage

The wheel ships SASS for every generation from Turing to Blackwell, plus a PTX
fallback for future GPUs — the CMake equivalent of:

```
-gencode arch=compute_75,code=sm_75      Turing    (RTX 20xx, T4)
-gencode arch=compute_80,code=sm_80      Ampere    (A100)
-gencode arch=compute_86,code=sm_86      Ampere    (RTX 30xx, A40)
-gencode arch=compute_89,code=sm_89      Ada       (RTX 40xx, L40)
-gencode arch=compute_90,code=sm_90      Hopper    (H100)
-gencode arch=compute_100,code=sm_100    Blackwell (B100/B200)
-gencode arch=compute_120,code=sm_120    Blackwell (RTX 50xx)
-gencode arch=compute_120,code=compute_120   PTX fallback
```

`sm_100`/`sm_120` need CUDA ≥ 12.8; on older toolkits CMake drops them and
ships `compute_90` PTX as the forward-compat fallback instead (with a warning).

## manylinux wheels (Linux only, cp38+)

```bash
pipx run cibuildwheel --platform linux
```

`[tool.cibuildwheel]` in `pyproject.toml` does the rest:

- **Image**: `manylinux_2_28` (manylinux2014's toolchain is too old for CUDA 12.8).
- **Toolkit**: `before-all` installs `cuda-nvcc-12-8` + `cuda-cudart-devel-12-8`
  from NVIDIA's RHEL 8 repo inside the container — nothing needed on the host.
- **One wheel per CPython version** 3.8 → latest, x86_64, musllinux skipped.

### Library bundling policy

- **`libcudart.so.12` — bundled.** The extension links the *shared* CUDA
  runtime (`CUDA_RUNTIME_LIBRARY Shared`) and `auditwheel repair` grafts the
  exact `libcudart` matching the build's CUDA version into the wheel.
  Target machines need no CUDA toolkit installed.
- **`libcuda.so` — never bundled.** It is the driver library, provided by the
  NVIDIA driver on the target node, and must resolve at runtime against
  whatever driver that node runs — hence
  `auditwheel repair --exclude libcuda.so --exclude libcuda.so.1`.
  (libcudart `dlopen`s it lazily, so wheels also import fine on GPU-less
  machines; only the CUDA path requires a driver.)
