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
#include "PyArray.h"
#include "LocalDomainStruct.h"

#ifndef MODULE_NAME
# define MODULE_NAME "manapy_part32"
#endif

// Convert macro to string
#define _STR(x) #x
#define STR(x) _STR(x)

#ifndef FLOAT_TYPE
  #define FLOAT_TYPE NPY_FLOAT32
#endif

#ifndef FDX_T
  #define FDX_T npy_float32
#endif

typedef FDX_T fdx_t;
const int float_type = FLOAT_TYPE;
const int int_type = NPY_INT32;


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
void time_it(const std::string &);

PyObject * create_local_domains(
LocalDomainStruct *local_domains,
PyArray<int32_t, 1> *part_vert,
PyArray<int32_t, 2> *node_cellid,
PyArray<int32_t, 2> *node_phyid,
PyArray<int32_t, 2> *cells,
PyArray<int8_t, 1> *cells_type,
PyArray<double, 2> *nodes,
PyArray<int32_t, 2> *phy_faces,
PyArray<int32_t, 1> *phy_faces_name,
int32_t nb_parts);

void devide(
    PyArray<int32_t, 1> *part_vert,
    PyArray<int32_t, 2> *node_cellid,
    PyArray<double, 2> *nodes,
    PyArray<int32_t, 2> *cells,
    PyArray<int8_t, 1> *cells_type,
    int32_t nb_parts
    );

#endif //MANAPY_PART_H
