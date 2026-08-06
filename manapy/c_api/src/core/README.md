# Adding a new kernel `<kernel>`

1. Create `common/<kernel>_common.hpp`
   - one `MANAPY_COMPUTE_HOST_DEVICE` function, e.g. `<kernel>_element(index_t i, ...)`

2. Create `cpu/<kernel>_cpu.cpp`
   - `#include "variable_compute.hpp"`
   - `#include "common/<kernel>_common.hpp"`
   - define `void <kernel>(...)` looping over all elements, calling `<kernel>_element`

3. Create `gpu/<kernel>_cuda.cu`
   - `#include "common/<kernel>_common.hpp"`
   - `#include "variable_compute.cuh"`
   - define `__global__ void <kernel>_kernel(...)` (grid-stride loop), calling `<kernel>_element`
   - define `void launch_<kernel>(..., cudaStream_t stream)` that launches `<kernel>_kernel`

4. Edit `variable_compute.hpp`
   - add the declaration of `void <kernel>(...)` (same signature as step 2)

5. Edit `variable_compute.cuh`
   - add the declaration of `void launch_<kernel>(..., cudaStream_t stream)` (same signature as step 3)

6. Create `bindings/<kernel>.cpp`
   - `#include "manapy_compute_types.hpp"`, `<cuda_runtime_api.h>`, `"bindings/registry.hpp"`, `"variable_compute.cuh"`, `"variable_compute.hpp"`
   - define `<kernel>_py(...)` taking `CFVec`/`CFMat`/`CIVec`/`CIMat`/`FVec` args, calling `<kernel>(...)` via `make_view<...>`
   - define `<kernel>_cuda_py(...)` taking `DCFVec`/`DCFMat`/`DCIVec`/`DCIMat`/`DFVec` args, calling `launch_<kernel>(...)`, checking `cudaGetLastError`/`cudaDeviceSynchronize`
   - define `void register_<kernel>(nb::module_ &m)` with `m.def("<kernel>", ...)` and `m.def("<kernel>_cuda", ...)`

7. Edit `bindings/registry.hpp`
   - add `void register_<kernel>(nb::module_ &m);`

8. Edit `bindings/module.cpp`
   - add `register_<kernel>(m);` inside `NB_MODULE(_core, m)`

9. Edit `CMakeLists.txt`
   - add `src/core/bindings/<kernel>.cpp`, `src/core/cpu/<kernel>_cpu.cpp`, `src/core/gpu/<kernel>_cuda.cu` to the `nanobind_add_module(...)` source list

10. Edit all four `python/manapy_compute_*/core/__init__.py`
    - add `<kernel>`, `<kernel>_cuda` to the `from ._core import (...)` block and to `__all__`

11. Rebuild
    - `cmake --build build --target _gradcore_32_32 _gradcore_32_64 _gradcore_64_32 _gradcore_64_64`

12. Reinstall if not already editable
    - `pip install -e . --no-build-isolation`
