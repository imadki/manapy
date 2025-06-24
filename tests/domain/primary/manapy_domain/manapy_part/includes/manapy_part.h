#ifndef MANAPY_PART_H
#define MANAPY_PART_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <tuple>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <metis.h>
#include <map>
#include <set>
#include <algorithm>
#include <stdarg.h>

typedef npy_float32 fdx_t;
constexpr int int_type = NPY_INT32;
constexpr int float_type = NPY_FLOAT32;

enum CELL_TYPE {
    Triangle = 1,
    Quad = 2,
    Tetra = 3,
    Hexahedron = 4,
    Pyramid = 5
};

std::vector<idx_t>    get_max_info(const idx_t cell_type) ;
int binary_search(const idx_t *array, idx_t item, idx_t size);
void    intersect_nodes(const idx_t *face_nodes, const idx_t nb_face_nodes, PyArrayObject *node_cellid,  idx_t *intersect);
void print_instant(const char *fmt, ...);

#endif //MANAPY_PART_H
