# METIS + GKlib, built from the vendored sources under this directory.
#
# METIS is compiled FOUR times rather than the two you would expect from its
# integer width alone. The reason is metis.h itself:
#
#     #if REALTYPEWIDTH == 32
#       typedef float real_t;
#     #elif REALTYPEWIDTH == 64
#       typedef double real_t;
#
# That `real_t` is a *global* typedef, and src/base/precision.hpp declares its
# own global `using real_t = float/double`. Every src/partitioning translation
# unit includes both headers, so the two widths have to agree or the build
# fails with a conflicting-declaration error. Hence one static library per
# (index width, float width) pair, named to match the precision pair of the
# manapy_compute_<float bits>_<int bits> package that links it:
#
#     metis_i32_f32  ->  manapy_compute_32_32
#     metis_i32_f64  ->  manapy_compute_64_32
#     metis_i64_f32  ->  manapy_compute_32_64
#     metis_i64_f64  ->  manapy_compute_64_64
#
# METIS's own real_t is only ever used for the tpwgts/ubvec arguments, which
# src/partitioning always passes as nullptr, so widening it is behaviourally
# inert -- it exists purely to keep the typedef consistent.

set(GKLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/GKlib")
set(METIS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/METIS")

file(GLOB GKLIB_SOURCES "${GKLIB_PATH}/src/*.c")
file(GLOB METIS_SOURCES "${METIS_PATH}/libmetis/*.c")

# METIS's programs/ (the gpmetis CLI and friends) and utils/ are deliberately
# not vendored: only libmetis/ is compiled into the extension modules.

function(add_metis_variant tgt ibits fbits)
  add_library(${tgt} STATIC ${GKLIB_SOURCES} ${METIS_SOURCES})

  target_compile_definitions(${tgt} PUBLIC
    IDXTYPEWIDTH=${ibits}
    REALTYPEWIDTH=${fbits})

  target_include_directories(${tgt} PUBLIC
    "${GKLIB_PATH}/include"
    "${METIS_PATH}/include"
    "${METIS_PATH}/libmetis")

  set_target_properties(${tgt} PROPERTIES
    # Linked into a shared Python extension module.
    POSITION_INDEPENDENT_CODE ON
    # Keep METIS/GKlib symbols out of each _core.so's dynamic symbol table.
    # Python dlopens extensions RTLD_LOCAL, so importing e.g.
    # manapy_compute_32_32.partitioning and manapy_compute_64_64.partitioning
    # in one interpreter already cannot cross-bind their two differently-sized
    # METIS_PartGraphKway; hiding the symbols makes that structural rather than
    # dependent on the loader flags.
    C_VISIBILITY_PRESET hidden)

  # Vendored upstream C89-era code: do not inherit the project's warning or
  # C++ settings, and silence the noisiest of what it does emit.
  target_compile_options(${tgt} PRIVATE
    -O3
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-unused-result)
endfunction()

add_metis_variant(metis_i32_f32 32 32)
add_metis_variant(metis_i32_f64 32 64)
add_metis_variant(metis_i64_f32 64 32)
add_metis_variant(metis_i64_f64 64 64)
