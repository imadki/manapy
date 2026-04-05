cmake_minimum_required(VERSION 3.15...4.0)
project(manapyGUI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(extern/pybind11)
add_subdirectory(extern/glfw)

find_package(Vulkan REQUIRED)
find_package(gmsh REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE IMGUI_SRC extern/imgui/*.cpp)
pybind11_add_module(manapyGUI ${SRC_FILES} ${IMGUI_SRC})


target_compile_definitions(manapyGUI PRIVATE
    SHADERS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

target_include_directories(manapyGUI PRIVATE
    extern/glm
    extern/imgui
)

target_link_libraries(manapyGUI PRIVATE 
    Vulkan::Vulkan
    glfw
    gmsh
)


set_target_properties(manapyGUI PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/manapy/gui
)

