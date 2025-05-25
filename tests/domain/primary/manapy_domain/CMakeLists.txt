cmake_minimum_required(VERSION 3.31)
project(manapy_domain)

set(CMAKE_CXX_STANDARD 14)

# python3 -m pybind11 --includes
# np.get_include()
INCLUDE_DIRECTORIES(
        "/home/aben-ham/anaconda3/envs/work/include/python3.8",
        "/home/aben-ham/anaconda3/envs/work/lib/python3.8/site-packages/pybind11/include",
        "/home/aben-ham/anaconda3/envs/work/lib/python3.8/site-packages/numpy/core/include"
)

add_executable(manapy_domain
        manapy_part/create_local_domain.cpp
        manapy_part/manapy_domain.c
        metis/main.c)
