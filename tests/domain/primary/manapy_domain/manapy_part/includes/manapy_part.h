#ifndef MANAPY_PART_H
#define MANAPY_PART_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <tuple>
#include <vector>
#include <Python.h>
#include <metis.h>
#include <map>
#include <set>
#include <algorithm>
#include <stdarg.h>
#include <thread>
#include "PyArray.h"
#include "LocalDomainStruct.h"
#include "Types.h"


enum CELL_TYPE {
    Triangle = 1,
    Quad = 2,
    Tetra = 3,
    Hexahedron = 4,
    Pyramid = 5
};

//utils.cpp
int32_t binary_search(const PyArray<int32_t, 1> &arr, int32_t item);
void intersect_arr(PyArray<int32_t, 2> *arr, PyArray<int32_t, 1> *indices, int32_t size, std::vector<int32_t> &intersect_arr);
std::vector<int32_t> get_max_info(int32_t cell_type);
void print_instant(const char *fmt, ...);
void time_it(const std::string &);

//compute_cell_center_volume.cpp
void compute_cell_center_area_2d(PyArray<int32_t, 2> const *cells, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 1> *cell_area, PyArray<fdx_t, 2> *cell_center);
void compute_cell_center_volume_3d(PyArray<int32_t, 2> const *cells, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 1> *cell_volume, PyArray<fdx_t, 2> *cell_center);
void compute_halo_cell_center_area_2d(PyArray<int32_t, 2> const *halo_halosext, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 2> *halo_centvol);
void compute_halo_cell_center_volume_3d(PyArray<int32_t, 2> const *halo_halosext, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 2> *halo_centvol);

//partitioning.cpp
PyObject * create_sub_domains(
    LocalDomainStruct *ld,
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<fdx_t, 2> *nodes,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    PyArray<int32_t, 2> *phy_faces,
    PyArray<int32_t, 1> *phy_faces_name,
    PyArray<int32_t, 2> *node_phyid,
    int32_t nb_parts
    );

#endif //MANAPY_PART_H
