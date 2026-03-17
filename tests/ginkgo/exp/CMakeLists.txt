cmake_minimum_required(VERSION 3.16)

project(ginkgo_solver)

set(GINKGO_INSTALL_FOLDER "/home/aben-ham/local")
find_package(Ginkgo 1.11.0 REQUIRED PATHS ${GINKGO_INSTALL_FOLDER})

set(CMAKE_CXX_COMPILER /usr/bin/c++)
#find_package(Ginkgo 1.11.0 REQUIRED)

add_executable(ginkgo_solver ginkgo.cpp)
target_link_libraries(ginkgo_solver Ginkgo::ginkgo)


file(COPY data/small_data DESTINATION .)
file(COPY data/large_data DESTINATION .)
file(COPY data/med_data DESTINATION .)
