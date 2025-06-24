cmake_minimum_required(VERSION 3.31)
project(manapy_domain)

set(CMAKE_CXX_STANDARD 14)

# python3 -m pybind11 --includes
# np.get_include()
INCLUDE_DIRECTORIES(
        "/home/aben-ham/anaconda3/envs/work/include/python3.8",
        "/home/aben-ham/anaconda3/envs/work/lib/python3.8/site-packages/pybind11/include",
        "/home/aben-ham/anaconda3/envs/work/lib/python3.8/site-packages/numpy/core/include",
        "/home/aben-ham/local/include",
        "manapy_part/includes"
)



add_executable(manapy_domain
        manapy_part/src/py_manapy_part.cpp
        metis/main.c
        manapy_part/src/utils.cpp
        manapy_part/includes/manapy_part.h
)
