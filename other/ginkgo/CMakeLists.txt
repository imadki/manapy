cmake_minimum_required(VERSION 3.16)

project(lu_solver)

set(GINKGO_INSTALL_FOLDER "/home/aben-ham/local")
find_package(Ginkgo 1.11.0 REQUIRED PATHS ${GINKGO_INSTALL_FOLDER})

add_executable(lu_solver lu_solver.cpp)
add_executable(example_lu_solver 1.cpp)
add_executable(ginkgo exp/ginkgo.cpp)

target_link_libraries(lu_solver Ginkgo::ginkgo)
target_link_libraries(example_lu_solver Ginkgo::ginkgo)
target_link_libraries(ginkgo Ginkgo::ginkgo)
