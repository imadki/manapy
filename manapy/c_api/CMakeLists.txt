cmake_minimum_required(VERSION 3.26)

# Resolve the CUDA toolkit before enabling the CUDA language. On machines
# with more than one CUDA install (e.g. a distro's nvidia-cuda-toolkit
# package alongside NVIDIA's official installer at /usr/local/cuda), CMake's
# default CUDA-language search can pick an nvcc from one toolkit whose
# sibling ptxas then resolves through PATH to a *different*, incompatible
# toolkit version (breaks with "Value 'smXX' is not defined for option
# 'gpu-name'" during compiler identification, before MANAPY_COMPUTE_CUDA_ARCHS
# below is even consulted). FindCUDAToolkit works without the CUDA language
# enabled and does a more careful search (CUDAToolkit_ROOT, then
# /usr/local/cuda, etc.), so pin CMAKE_CUDA_COMPILER to its nvcc to keep nvcc
# and ptxas from the same, self-consistent installation.
find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "")
endif()

# C is needed for the vendored METIS/GKlib sources under third_party/.
project(manapy_compute LANGUAGES C CXX CUDA)

# C++20: ArrayView (src/base/array_view.hpp) uses requires-clauses.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
# CUDAToolkit was already located above, before project(), to pin CMAKE_CUDA_COMPILER.

# Optional: powers create_cellfid's `#pragma omp parallel for` (src/domain).
# Without it, that pragma is just ignored by the compiler and the loop runs
# sequentially -- still correct, just single-threaded.
find_package(OpenMP COMPONENTS CXX)

# Locate the nanobind CMake package installed as a build requirement.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Vendored METIS/GKlib: four static libraries, one per precision pair, for the
# partitioning module. EXCLUDE_FROM_ALL so a variant is only built when a
# target actually links it.
add_subdirectory(third_party EXCLUDE_FROM_ALL)

# One SASS binary per supported GPU generation, equivalent to:
#   -gencode arch=compute_75,code=sm_75    Turing   (RTX 20xx, T4)
#   -gencode arch=compute_80,code=sm_80    Ampere   (A100)
#   -gencode arch=compute_86,code=sm_86    Ampere   (RTX 30xx, A40)
#   -gencode arch=compute_89,code=sm_89    Ada      (RTX 40xx, L40)
#   -gencode arch=compute_90,code=sm_90    Hopper   (H100)
#   -gencode arch=compute_100,code=sm_100  Blackwell (B100/B200)
#   -gencode arch=compute_120,code=sm_120  Blackwell (RTX 50xx)
#   -gencode arch=compute_120,code=compute_120   PTX fallback for newer GPUs
set(MANAPY_COMPUTE_CUDA_ARCHS 75-real 80-real 86-real 89-real 90-real)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  # sm_100 / sm_120 only exist from CUDA 12.8 on.
  list(APPEND MANAPY_COMPUTE_CUDA_ARCHS 100-real 120-real 120-virtual)
else()
  message(WARNING
    "CUDA ${CMAKE_CUDA_COMPILER_VERSION} < 12.8: Blackwell targets "
    "(sm_100/sm_120) skipped; shipping compute_90 PTX as forward-compat fallback.")
  list(APPEND MANAPY_COMPUTE_CUDA_ARCHS 90-virtual)
endif()

# ---------------------------------------------------------------------------
# Cell-gradient kernel: one module per precision pair, packaged as the `core`
# submodule of manapy_compute_<float bits>_<int bits> (e.g. manapy_compute_32_64
# = float32 data, int64 indices). Each package ships its own core/_core built
# from the same real_t/index_t source, recompiled once per precision pair. The
# domain kernel below shares the same top-level package as its `domain`
# submodule.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _gradcore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/core/bindings/module.cpp
      src/core/bindings/cell_gradient_2d.cpp
      src/core/bindings/center_to_vertex_2d.cpp
      src/core/bindings/face_gradient_2d.cpp
      src/core/bindings/barthlimiter_2d.cpp
      src/core/bindings/vanalbadalimiter_2d.cpp
      src/core/bindings/face_gradient_3d.cpp
      src/core/bindings/cell_gradient_3d.cpp
      src/core/bindings/center_to_vertex_3d.cpp
      src/core/bindings/barthlimiter_3d.cpp
      src/core/bindings/vanalbadalimiter_3d.cpp
      src/core/bindings/facetocell.cpp
      src/core/bindings/celltoface.cpp
      src/core/cpu/cell_gradient_2d_cpu.cpp
      src/core/cpu/center_to_vertex_2d_cpu.cpp
      src/core/cpu/face_gradient_2d_cpu.cpp
      src/core/cpu/barthlimiter_2d_cpu.cpp
      src/core/cpu/vanalbadalimiter_2d_cpu.cpp
      src/core/cpu/face_gradient_3d_cpu.cpp
      src/core/cpu/cell_gradient_3d_cpu.cpp
      src/core/cpu/center_to_vertex_3d_cpu.cpp
      src/core/cpu/barthlimiter_3d_cpu.cpp
      src/core/cpu/vanalbadalimiter_3d_cpu.cpp
      src/core/cpu/facetocell_cpu.cpp
      src/core/cpu/celltoface_cpu.cpp
      src/core/gpu/cell_gradient_2d_cuda.cu
      src/core/gpu/center_to_vertex_2d_cuda.cu
      src/core/gpu/face_gradient_2d_cuda.cu
      src/core/gpu/barthlimiter_2d_cuda.cu
      src/core/gpu/vanalbadalimiter_2d_cuda.cu
      src/core/gpu/face_gradient_3d_cuda.cu
      src/core/gpu/cell_gradient_3d_cuda.cu
      src/core/gpu/center_to_vertex_3d_cuda.cu
      src/core/gpu/barthlimiter_3d_cuda.cu
      src/core/gpu/vanalbadalimiter_3d_cuda.cu
      src/core/gpu/facetocell_cuda.cu
      src/core/gpu/celltoface_cuda.cu)
    target_include_directories(${tgt} PRIVATE src/base src/core)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    # Every package imports its module as <pkg>.core._core, so all four
    # targets share the output name _core and get their own output directory
    # (nested under core/, alongside the domain kernel's domain/ below) to
    # avoid clashing in the build tree. The gradient kernel is compiled for
    # the same GPU generations as the main _core module and links the shared
    # CUDA runtime so auditwheel can graft libcudart into the wheel.
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/core"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/core)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Domain kernel: mesh connectivity/geometry, ported from
# src/domain/to_convert.py (see src/domain/Steps.md). CPU-only, and packaged
# as the `domain` submodule of the same manapy_compute_<float bits>_<int bits>
# package the gradient kernel above ships as `core` -- one _core module per
# precision pair. Sources get added here batch by batch as src/domain/Steps.md
# is worked through; only bindings/module.cpp exists so far (Batch 0 has no
# bindings).
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _domaincore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/domain/bindings/module.cpp
      src/domain/bindings/count_max_node_cellid.cpp
      src/domain/bindings/create_node_cellid.cpp
      src/domain/bindings/get_cell_nb_phyid.cpp
      src/domain/bindings/count_max_cell_cellnid.cpp
      src/domain/bindings/create_cell_cellnid.cpp
      src/domain/bindings/create_info.cpp
      src/domain/bindings/compute_face_info_2d.cpp
      src/domain/bindings/compute_face_info_3d.cpp
      src/domain/bindings/create_bf_cellid.cpp
      src/domain/bindings/create_ghost_info.cpp
      src/domain/bindings/create_ghost_tables.cpp
      src/domain/bindings/count_max_bcell_halophyid.cpp
      src/domain/bindings/create_bcell_halophyid.cpp
      src/domain/bindings/get_max_b_ncellid.cpp
      src/domain/bindings/create_b_ncellid.cpp
      src/domain/bindings/create_halo_ghost_tables.cpp
      src/domain/bindings/create_cellfid.cpp
      src/domain/bindings/define_node_oldname.cpp
      src/domain/bindings/define_face_name.cpp
      src/domain/bindings/create_halo_cells.cpp
      src/domain/bindings/face_gradient_info_2d.cpp
      src/domain/bindings/face_gradient_info_3d.cpp
      src/domain/bindings/fv_face_geometry.cpp
      src/domain/bindings/variables_2d.cpp
      src/domain/bindings/variables_3d.cpp
      src/domain/bindings/create_normal_face_of_cell.cpp
      src/domain/bindings/dist_ortho_function_2d.cpp
      src/domain/bindings/pair_periodic_faces.cpp
      src/domain/bindings/node_periodic_bits.cpp
      src/domain/bindings/accum_periodic_dir.cpp
      src/domain/bindings/compute_cell_center_volume.cpp
      src/domain/cpu/count_max_node_cellid_cpu.cpp
      src/domain/cpu/create_node_cellid_cpu.cpp
      src/domain/cpu/get_cell_nb_phyid_cpu.cpp
      src/domain/cpu/count_max_cell_cellnid_cpu.cpp
      src/domain/cpu/create_cell_cellnid_cpu.cpp
      src/domain/cpu/create_info_cpu.cpp
      src/domain/cpu/compute_face_info_2d_cpu.cpp
      src/domain/cpu/compute_face_info_3d_cpu.cpp
      src/domain/cpu/create_bf_cellid_cpu.cpp
      src/domain/cpu/create_ghost_info_cpu.cpp
      src/domain/cpu/create_ghost_tables_cpu.cpp
      src/domain/cpu/count_max_bcell_halophyid_cpu.cpp
      src/domain/cpu/create_bcell_halophyid_cpu.cpp
      src/domain/cpu/get_max_b_ncellid_cpu.cpp
      src/domain/cpu/create_b_ncellid_cpu.cpp
      src/domain/cpu/create_halo_ghost_tables_cpu.cpp
      src/domain/cpu/create_cellfid_cpu.cpp
      src/domain/cpu/define_node_oldname_cpu.cpp
      src/domain/cpu/define_face_name_cpu.cpp
      src/domain/cpu/create_halo_cells_cpu.cpp
      src/domain/cpu/face_gradient_info_2d_cpu.cpp
      src/domain/cpu/face_gradient_info_3d_cpu.cpp
      src/domain/cpu/fv_face_geometry_cpu.cpp
      src/domain/cpu/variables_2d_cpu.cpp
      src/domain/cpu/variables_3d_cpu.cpp
      src/domain/cpu/create_normal_face_of_cell_cpu.cpp
      src/domain/cpu/dist_ortho_function_2d_cpu.cpp
      src/domain/cpu/pair_periodic_faces_cpu.cpp
      src/domain/cpu/node_periodic_bits_cpu.cpp
      src/domain/cpu/accum_periodic_dir_cpu.cpp
      src/domain/cpu/compute_cell_center_volume_cpu.cpp)
    target_include_directories(${tgt} PRIVATE src/base src/domain)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE -O3)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/domain")

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/domain)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Partitioning kernels: METIS-backed domain decomposition, ported from manapy's
# c_api. CPU-only, packaged as the `partitioning` submodule of the same
# manapy_compute_<float bits>_<int bits> package as `core`, `domain` and
# `boundary` -- one _core module per precision pair.
#
# Each target links the METIS variant built for its own precision pair:
# IBITS picks IDXTYPEWIDTH (so METIS's idx_t is index_t) and FBITS picks
# REALTYPEWIDTH (so METIS's real_t typedef agrees with src/base/precision.hpp's
# -- they collide otherwise, which is why there are four METIS builds; see
# third_party/CMakeLists.txt).
#
# src/domain/cpu/compute_cell_center_volume_cpu.cpp is compiled in here as well
# as into the domain target: create_halos calls its halo_* entry points to fill
# each subdomain's halo_centvol.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _partitioningcore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/partitioning/bindings/module.cpp
      src/partitioning/bindings/metis_partitioning.cpp
      src/partitioning/bindings/create_local_domains.cpp
      src/partitioning/src/partitioning.cpp
      src/partitioning/src/metis_partitioning.cpp
      src/partitioning/src/local_domain_struct.cpp
      src/partitioning/src/utils.cpp
      src/domain/cpu/compute_cell_center_volume_cpu.cpp)
    target_include_directories(${tgt} PRIVATE
      src/base src/partitioning src/partitioning/includes src/domain)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE -O3)
    target_link_libraries(${tgt} PRIVATE metis_i${IBITS}_f${FBITS})
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/partitioning")

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/partitioning)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Boundary-condition kernels: ghost / halo-ghost values for the scalar
# conditions (dirichlet, neumann, neumannNH, nonslip) and the vector free-slip
# wall, ported from src/boundary/to_convert.py. CPU + CUDA, packaged as the
# `boundary` submodule of the same manapy_compute_<float bits>_<int bits>
# package as `core` and `domain` -- one _core module per precision pair, built
# GPU-capable like `core`: compiled for every supported GPU generation and
# linked against the shared CUDA runtime.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _boundarycore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/boundary/bindings/module.cpp
      src/boundary/bindings/ghost_value.cpp
      src/boundary/bindings/haloghost_value.cpp
      src/boundary/bindings/ghost_value_slip_2d.cpp
      src/boundary/bindings/ghost_value_slip_3d.cpp
      src/boundary/bindings/haloghost_value_slip_2d.cpp
      src/boundary/bindings/haloghost_value_slip_3d.cpp
      src/boundary/cpu/ghost_value_cpu.cpp
      src/boundary/cpu/haloghost_value_cpu.cpp
      src/boundary/cpu/ghost_value_slip_2d_cpu.cpp
      src/boundary/cpu/ghost_value_slip_3d_cpu.cpp
      src/boundary/cpu/haloghost_value_slip_2d_cpu.cpp
      src/boundary/cpu/haloghost_value_slip_3d_cpu.cpp
      src/boundary/gpu/ghost_value_cuda.cu
      src/boundary/gpu/haloghost_value_cuda.cu
      src/boundary/gpu/ghost_value_slip_2d_cuda.cu
      src/boundary/gpu/ghost_value_slip_3d_cuda.cu
      src/boundary/gpu/haloghost_value_slip_2d_cuda.cu
      src/boundary/gpu/haloghost_value_slip_3d_cuda.cu)
    target_include_directories(${tgt} PRIVATE src/base src/boundary)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/boundary"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/boundary)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Advection (advec) solver kernels: explicit finite-volume solver ported from
# the manapy advection solver (src/solvers). CPU + CUDA, packaged as the nested
# manapy_compute_<float bits>_<int bits>.solvers.advec submodule (alongside
# solvers.utils) -- one _core module per precision pair. Built like `core`
# (GPU-capable): compiled for every supported GPU generation and linked against
# the shared CUDA runtime. Sources get added here as more of the solver is
# ported.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _adveccore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/solvers/bindings/advec/module.cpp
      src/solvers/bindings/advec/explicitscheme_convective_2d.cpp
      src/solvers/bindings/advec/explicitscheme_convective_3d.cpp
      src/solvers/bindings/advec/time_step.cpp
      src/solvers/cpu/advec/explicitscheme_convective_2d_cpu.cpp
      src/solvers/cpu/advec/explicitscheme_convective_3d_cpu.cpp
      src/solvers/cpu/advec/time_step_cpu.cpp
      src/solvers/gpu/advec/explicitscheme_convective_2d_cuda.cu
      src/solvers/gpu/advec/explicitscheme_convective_3d_cuda.cu
      src/solvers/gpu/advec/time_step_cuda.cu)
    target_include_directories(${tgt} PRIVATE
      src/base src/solvers src/solvers/headers/advec)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/solvers/advec"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/solvers/advec)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Advection-diffusion (advecdiff) solver kernels: ported from
# src/solvers/to_convert.py (see src/solvers/Steps.md). CPU + CUDA, packaged as
# the nested manapy_compute_<float bits>_<int bits>.solvers.advecdiff submodule
# -- one _core module per precision pair, built like advec (GPU-capable).
# Sources get added here batch by batch; Batch 1 scaffolds an empty module.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _advecdiffcore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/solvers/bindings/advecdiff/module.cpp
      src/solvers/bindings/advecdiff/explicitscheme_convective_2d.cpp
      src/solvers/bindings/advecdiff/explicitscheme_convective_3d.cpp
      src/solvers/bindings/advecdiff/explicitscheme_dissipative.cpp
      src/solvers/bindings/advecdiff/time_step.cpp
      src/solvers/cpu/advecdiff/explicitscheme_convective_2d_cpu.cpp
      src/solvers/cpu/advecdiff/explicitscheme_convective_3d_cpu.cpp
      src/solvers/cpu/advecdiff/explicitscheme_dissipative_cpu.cpp
      src/solvers/cpu/advecdiff/time_step_cpu.cpp
      src/solvers/gpu/advecdiff/explicitscheme_convective_2d_cuda.cu
      src/solvers/gpu/advecdiff/explicitscheme_convective_3d_cuda.cu
      src/solvers/gpu/advecdiff/explicitscheme_dissipative_cuda.cu
      src/solvers/gpu/advecdiff/time_step_cuda.cu)
    target_include_directories(${tgt} PRIVATE
      src/base src/solvers src/solvers/headers/advecdiff)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/solvers/advecdiff"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/solvers/advecdiff)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Pure-diffusion (diffusion) solver kernels: ported from
# src/solvers/to_convert.py. The dissipative flux plus a diffusion-only CFL
# time step; the forward-Euler update is the shared solvers.utils one. CPU +
# CUDA, packaged as the nested
# manapy_compute_<float bits>_<int bits>.solvers.diffusion submodule -- one
# _core module per precision pair, built like advec (GPU-capable).
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _diffusioncore_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/solvers/bindings/diffusion/module.cpp
      src/solvers/bindings/diffusion/explicitscheme_dissipative.cpp
      src/solvers/bindings/diffusion/time_step.cpp
      src/solvers/cpu/diffusion/explicitscheme_dissipative_cpu.cpp
      src/solvers/cpu/diffusion/time_step_cpu.cpp
      src/solvers/gpu/diffusion/explicitscheme_dissipative_cuda.cu
      src/solvers/gpu/diffusion/time_step_cuda.cu)
    target_include_directories(${tgt} PRIVATE
      src/base src/solvers src/solvers/headers/diffusion)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/solvers/diffusion"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/solvers/diffusion)
  endforeach()
endforeach()

# ---------------------------------------------------------------------------
# Solver utilities: kernels common to all solvers (advec, advecDiff, ...)
# rather than to any one of them -- e.g. initial-condition setup and the
# forward-Euler cell-field update. Packaged as the nested
# manapy_compute_<float bits>_<int bits>.solvers.utils submodule. Mixed CPU +
# GPU (per-function: the Gaussian init kernels are CPU-only, update_new_value
# has a GPU counterpart), so built GPU-capable like advec -- compiled for every
# supported GPU generation and linked against the shared CUDA runtime. One
# _core module per precision pair. Sources get added here as more shared
# utilities are ported.
# ---------------------------------------------------------------------------
foreach(FBITS IN ITEMS 32 64)
  foreach(IBITS IN ITEMS 32 64)
    set(pkg manapy_compute_${FBITS}_${IBITS})
    set(tgt _solversutils_${FBITS}_${IBITS})

    nanobind_add_module(${tgt} NB_STATIC
      src/solvers/bindings/utils/module.cpp
      src/solvers/bindings/utils/initialisation_gaussian.cpp
      src/solvers/bindings/utils/update_new_value.cpp
      src/solvers/cpu/utils/initialisation_gaussian_2d_cpu.cpp
      src/solvers/cpu/utils/initialisation_gaussian_3d_cpu.cpp
      src/solvers/cpu/utils/update_new_value_cpu.cpp
      src/solvers/gpu/utils/update_new_value_cuda.cu)
    target_include_directories(${tgt} PRIVATE
      src/base src/solvers src/solvers/headers/utils)
    target_compile_definitions(${tgt} PRIVATE
      MANAPY_COMPUTE_FLOAT_BITS=${FBITS}
      MANAPY_COMPUTE_INT_BITS=${IBITS})
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>)
    set_target_properties(${tgt} PROPERTIES
      OUTPUT_NAME "_core"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pkg}/solvers/utils"
      CUDA_ARCHITECTURES "${MANAPY_COMPUTE_CUDA_ARCHS}"
      CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)

    install(TARGETS ${tgt} LIBRARY DESTINATION ${pkg}/solvers/utils)
  endforeach()
endforeach()
