cmake_minimum_required(VERSION 3.17)
project(manapy_part LANGUAGES C CXX)

# =========================================================
# Global settings
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Static linking (important for Python wheels)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SHARED OFF CACHE BOOL "" FORCE)

# =========================================================
# Python
# =========================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# =========================================================
# NumPy include path
# =========================================================
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NumPy_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy not found")
endif()

# =========================================================
# GKlib / METIS configuration
# =========================================================

# Add our custom third_party CMakeLists.txt that builds metis_32 (int32) and metis_64 (int64)
add_subdirectory(third_party EXCLUDE_FROM_ALL)

# =========================================================
# Source files
# =========================================================
set(SOURCES
        src/py_manapy_part.cpp
        src/partitioning.cpp
        src/utils.cpp
        src/LocalDomainStruct.cpp
        src/compute_cell_center_volume.cpp
)

# =========================================================
# Common include directories
# =========================================================
set(INCLUDE_DIRS
        ${NumPy_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/includes
)

# =========================================================
# Function to create Python module
# =========================================================
function(create_manapy_module target_name int_type float_type metis_target)
    python_add_library(${target_name} MODULE ${SOURCES})

    target_include_directories(${target_name} PRIVATE ${INCLUDE_DIRS})

    target_compile_definitions(${target_name} PRIVATE
            MODULE_NAME=${target_name}
            FLOAT_TYPE=${float_type}
            INT_TYPE=${int_type}
            NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -O3) #-Wall -Wextra
    endif()

    target_link_libraries(${target_name} PRIVATE
            ${metis_target}
    )

    # Remove lib prefix (Python expects module.so)
    set_target_properties(${target_name} PROPERTIES
            PREFIX ""
    )
endfunction()

# =========================================================
# Build modules
# =========================================================
create_manapy_module(manapy_part32_32 int32 float metis_32)
create_manapy_module(manapy_part32_64 int32 double metis_32)
create_manapy_module(manapy_part64_32 int64 float metis_64)
create_manapy_module(manapy_part64_64 int64 double metis_64)

# =========================================================
# Install (required for wheel packaging)
# =========================================================
install(TARGETS
        manapy_part32_32
        manapy_part32_64
        manapy_part64_32
        manapy_part64_64
        LIBRARY DESTINATION .
)