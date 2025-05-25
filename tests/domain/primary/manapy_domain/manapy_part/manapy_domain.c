/*
 * manapy_domain.c – Python C‑extension exposing
 *     make_n_part(graph: ndarray[int64], nb_parts: int)
 *
 * Graph format (dense w/ size sentinel)
 * -------------------------------------
 *   2‑D NumPy int64 array of shape (nvtxs, max_deg + 1)
 *   The last column in every row stores the integer `size` – i.e. how many
 *   neighbour entries in that row are valid.  The first `size` items are the
 *   0‑based neighbour vertex indices.  No sentinel value is needed.  Excess
 *   columns (if any) may contain arbitrary numbers and are ignored.
 *
 * Example (4 vertices):
 *   [[1, 2, 0, 2],   // v0 → {1,2}
 *    [0, 2, 3, 3],   // v1 → {0,2,3}
 *    [0, 1, 0, 2],   // v2 → {0,1}
 *    [1, 0, 0, 1]]   // v3 → {1}
 *
 * The code converts this dense form to CSR (xadj/adjncy), calls
 * METIS_PartGraphKway, and returns (edge_cut, part_vector) where part_vector
 * is a 1‑D NumPy int64 array of length nvtxs with values in 0..nb_parts‑1.
 *
 * Build (Unix):
 *   sudo apt-get install libmetis-dev
 *   python -m pip install numpy pybind11
 *   gcc -O2 -shared -fPIC $(python3 -m pybind11 --includes) \
 *       -I${METIS_INCLUDE_DIR:-/usr/include} \
 *       -L${METIS_LIB_DIR:-/usr/lib} -lmetis \
 *       manapy_domain.c -o manapy_domain$(python3-config --extension-suffix)
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <metis.h>
#include <stdlib.h>
#include <stdint.h>

/* -------- Utility: dense array → CSR ------------------------------ */
static int dense_to_csr(PyArrayObject *g,
                        idx_t **xadj, idx_t **adjncy,
                        idx_t *nvtxs, idx_t *total_deg)
{
    if (PyArray_NDIM(g) != 2 || PyArray_TYPE(g) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "graph must be a 2D int64 NumPy array");
        return -1;
    }
    npy_intp *dims = PyArray_DIMS(g);
    *nvtxs = (idx_t)dims[0];
    idx_t max_cols = (idx_t)dims[1];
    if (max_cols < 1) {
        PyErr_SetString(PyExc_ValueError, "graph must have at least one column (size field)");
        return -1;
    }

    idx_t deg_sum = 0;
    /* First pass – get total degree */
    for (idx_t v = 0; v < *nvtxs; ++v) {
        int64_t sz = *(int64_t *)PyArray_GETPTR2(g, v, max_cols - 1);
        if (sz < 0 || sz >= max_cols) {
            PyErr_Format(PyExc_ValueError,
                         "row %lld has invalid size %lld (max_cols=%lld)",
                         (long long)v, (long long)sz, (long long)(max_cols - 1));
            return -1;
        }
        deg_sum += (idx_t)sz;
    }

    *total_deg = deg_sum;

    *xadj   = (idx_t *)malloc((*nvtxs + 1) * sizeof(idx_t));
    *adjncy = (idx_t *)malloc(deg_sum       * sizeof(idx_t));
    PySys_WriteStdout("Number of adjncy => %d \n", deg_sum);
    if (!*xadj || !*adjncy) {
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return -1;
    }

    idx_t cursor = 0;
    (*xadj)[0] = 0;
    for (idx_t v = 0; v < *nvtxs; ++v) {
        int64_t sz = *(int64_t *)PyArray_GETPTR2(g, v, max_cols - 1);
        for (idx_t j = 0; j < (idx_t)sz; ++j) {
            int64_t nb = *(int64_t *)PyArray_GETPTR2(g, v, j);
            if (nb < 0 || nb >= *nvtxs) {
                free(*xadj); free(*adjncy);
                PyErr_Format(PyExc_ValueError,
                             "row %lld, col %lld has invalid neighbour %lld",
                             (long long)v, (long long)j, (long long)nb);
                return -1;
            }
            (*adjncy)[cursor++] = (idx_t)nb;
        }
        (*xadj)[v + 1] = cursor;
    }
    return 0;
}

/* -------- make_n_part(graph, nb_parts) ----------------------------- */
static PyObject *py_make_n_part(PyObject *self, PyObject *args)
{
    PyObject *graph_obj = NULL;
    int nb_parts;
    if (!PyArg_ParseTuple(args, "Oi", &graph_obj, &nb_parts))
        return NULL;
    if (nb_parts < 2) {
        PyErr_SetString(PyExc_ValueError, "nb_parts must be ≥ 2");
        return NULL;
    }

    PyArrayObject *g = (PyArrayObject *)PyArray_FROM_OTF(graph_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!g) return NULL;

    idx_t *xadj = NULL, *adjncy = NULL;
    idx_t nvtxs = 0, deg_sum = 0;
    PySys_WriteStdout("Start dense_to_csr\n");
    if (dense_to_csr(g, &xadj, &adjncy, &nvtxs, &deg_sum) < 0) {
        Py_DECREF(g);
        return NULL; /* error already set */
    }
    Py_DECREF(g);
    PySys_WriteStdout("End dense_to_csr\n");

    /* allocate partition vector in idx_t precision */
    idx_t *part_idx = (idx_t *)malloc(nvtxs * sizeof(idx_t));
    if (!part_idx) {
        free(xadj); free(adjncy);
        PyErr_SetString(PyExc_MemoryError, "malloc failed");
        return NULL;
    }

    idx_t ncon = 1, objval = 0;
    idx_t nparts_idx = (idx_t)nb_parts;

    int status = METIS_PartGraphKway(&nvtxs, &ncon,
                                     xadj, adjncy,
                                     NULL, NULL, NULL,
                                     &nparts_idx,
                                     NULL, NULL,
                                     NULL,
                                     &objval, part_idx);

    free(xadj); free(adjncy);

    if (status != METIS_OK) {
        free(part_idx);
        PyErr_Format(PyExc_RuntimeError, "METIS_PartGraphKway failed (status=%d)", status);
        return NULL;
    }

    npy_intp dims[1] = { (npy_intp)nvtxs };
    PyObject *part_array = NULL;

    if (sizeof(idx_t) == sizeof(int64_t)) {
        /* Directly wrap the idx_t buffer – it is already int64 */
        part_array = PyArray_SimpleNewFromData(1, dims, NPY_INT64, part_idx);
        if (!part_array) { free(part_idx); return NULL; }
        PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    } else {
        /* Convert 32‑bit idx_t → 64‑bit NumPy array */
        int64_t *buf64 = (int64_t *)malloc(nvtxs * sizeof(int64_t));
        if (!buf64) { free(part_idx); PyErr_SetString(PyExc_MemoryError, "malloc failed"); return NULL; }
        for (idx_t i = 0; i < nvtxs; ++i) buf64[i] = (int64_t)part_idx[i];
        free(part_idx);
        part_array = PyArray_SimpleNewFromData(1, dims, NPY_INT64, buf64);
        if (!part_array) { free(buf64); return NULL; }
        PyArray_ENABLEFLAGS((PyArrayObject *)part_array, NPY_ARRAY_OWNDATA);
    }

    PyObject *edge_cut_py = PyLong_FromLongLong((long long)objval);
    return Py_BuildValue("NO", edge_cut_py, part_array);
}

/* -------- module definition --------------------------------------- */
static PyMethodDef ManapyMethods[] = {
    { "make_n_part", py_make_n_part, METH_VARARGS,
      "Partition a dense graph using METIS.\n\n"
      "Parameters\n----------\n"
      "graph : ndarray[int64]  shape (nvtxs, max_deg+1)\n\n"
      "    The last column stores the neighbour count.\n"
      "nb_parts : int >= 2\n\n"
      "Returns\n-------\n"
      "edge_cut : int\n    Number of cut edges reported by METIS.\n\n"
      "partitions : ndarray[int64] shape (nvtxs,)\n    Vertex-to-partition mapping." },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef manapy_module = {
    PyModuleDef_HEAD_INIT,
    "manapy_domain",  /* m_name */
    "Manapy domain partitioning helpers (METIS-backed)", /* m_doc */
    -1,                /* m_size */
    ManapyMethods      /* m_methods */
};

PyMODINIT_FUNC PyInit_manapy_domain(void)
{
    import_array(); /* initialise NumPy C-API */
    return PyModule_Create(&manapy_module);
}
