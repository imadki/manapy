cmake_minimum_required(VERSION 3.17)

set(GKLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/GKlib")

file(GLOB GKLIB_SOURCES "${GKLIB_PATH}/src/*.c")
file(GLOB METIS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/METIS/libmetis/*.c")

# ---- metis_32 ----
add_library(metis_32 STATIC ${GKLIB_SOURCES} ${METIS_SOURCES})
target_compile_definitions(metis_32 PUBLIC
        IDXTYPEWIDTH=32
        REALTYPEWIDTH=32
)
target_include_directories(metis_32 PUBLIC
        "${GKLIB_PATH}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/METIS/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/METIS/libmetis"
)
set_target_properties(metis_32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- metis_64 ----
add_library(metis_64 STATIC ${GKLIB_SOURCES} ${METIS_SOURCES})
target_compile_definitions(metis_64 PUBLIC
        IDXTYPEWIDTH=64
        REALTYPEWIDTH=32
)
target_include_directories(metis_64 PUBLIC
        "${GKLIB_PATH}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/METIS/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/METIS/libmetis"
)
set_target_properties(metis_64 PROPERTIES POSITION_INDEPENDENT_CODE ON)
