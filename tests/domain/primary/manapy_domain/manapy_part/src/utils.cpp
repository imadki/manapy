#include "manapy_part.h"

std::vector<idx_t>    get_max_info(const idx_t cell_type) {
    if (cell_type == CELL_TYPE::Triangle) {
        return {3, 2, 3};
    } else if (cell_type == CELL_TYPE::Quad) {
        return {4, 2, 4};
    } else if (cell_type == CELL_TYPE::Tetra) {
        return {4, 3, 4};
    } else if (cell_type == CELL_TYPE::Hexahedron) {
        return {6, 4, 8};
    } else if (cell_type == CELL_TYPE::Pyramid) {
        return {5, 4, 5};
    }
    return {0, 0, 0};
}

/**
 * Performs binary search on a sorted array of integers.
 *
 * @param array  Pointer to array of integers.
 * @param item   The item to search for.
 * @param size   The size of the array.
 * @return       Index >= 0 if found, else -1.
 */
int binary_search(const idx_t *array, idx_t item, idx_t size) {
    idx_t left = 0;
    idx_t right = size - 1;

    while (left <= right) {
        const idx_t mid = (left + right) / 2;
        const idx_t mid_val = array[mid];

        if (mid_val == item) {
            return mid;
        } else if (mid_val < item) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

void    intersect_nodes(const idx_t *face_nodes, const idx_t nb_face_nodes, PyArrayObject *node_cellid,  idx_t *intersect) {
    idx_t index = 0;

    intersect[0] = -1;
    intersect[1] = -1;
    const npy_intp *dims = PyArray_DIMS(node_cellid);

    const idx_t *cells = (idx_t *)PyArray_GETPTR2(node_cellid, face_nodes[0], 0);
    for (idx_t i = 0; i < cells[dims[1] - 1]; i++) {
        intersect[index] = cells[i];
        for (idx_t j = 1; j < nb_face_nodes; j++) {
            const idx_t *n_cells = (idx_t *)PyArray_GETPTR2(node_cellid, face_nodes[j], 0);
            const idx_t size = n_cells[dims[1] - 1];
            if (binary_search(n_cells, cells[i], size) == -1) {
                intersect[index] = -1;
                break;
            }
        }
        if (intersect[index] != -1)
            index++;
        if (index >= 2)
            return;
    }
}

# define PRINT_DEBUG false
void print_instant(const char *fmt, ...) {
#ifdef PRINT_DEBUG
    char buffer[512];  // temp string buffer
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);  // format the string
    va_end(args);

    // Import sys module and write to sys.stdout
    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) return;

    PyObject *stdout = PyObject_GetAttrString(sys, "stdout");
    if (stdout) {
        PyObject *write_result = PyObject_CallMethod(stdout, "write", "s", buffer);
        Py_XDECREF(write_result);

        PyObject *flush_result = PyObject_CallMethod(stdout, "flush", NULL);
        Py_XDECREF(flush_result);

        Py_DECREF(stdout);
    }

    Py_DECREF(sys);
#endif
}