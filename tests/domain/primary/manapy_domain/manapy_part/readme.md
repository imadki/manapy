PyArrayObject
```c
int64_t *base = (int64_t *)PyArray_DATA(arr);
PyArray_NDIM(arr)
PyArray_TYPE(arr) != NPY_INT64
PyArray_Check(arr)
int64_t *elem = (int64_t *)PyArray_GETPTR2(arr, i, j); a pointer to one element 

--> contiguity
if (!PyArray_ISCARRAY(arr)) {
    PyErr_SetString(PyExc_ValueError, "Array must be contiguous and C-ordered");
    return NULL;
}

assert(PyArray_ISCARRAY(array));            // C-contiguous, aligned, writable
assert(PyArray_ISCONTIGUOUS(array));        // C-contiguous
assert(PyArray_ISWRITEABLE(array));         // Writable
assert(PyArray_ISOWNDATA(array));           // Owns its data


--> Create an array
auto res = PyArray_ZEROS(1, (npy_intp[]){5}, NPY_INT64, 0);
if (!res) {
    return NULL;
    Py_XDECREF(res);
}
```

```c
create a tuple
PyObject *t = Py_BuildValue("(is)", 42, "hello");

Second method
PyObject *tuple = PyTuple_New(2);                   // create empty tuple of size 2
PyTuple_SET_ITEM(tuple, 0, PyLong_FromLong(42));    // sets index 0 to int(42)
PyTuple_SET_ITEM(tuple, 1, PyUnicode_FromString("hello"));  // sets index 1
return tuple;



```

## Reference count

🔁 Understanding Reference Counting in the Python C API
In the Python C API, reference counting is the core memory management model. Every PyObject* keeps track of how many references point to it. When the count drops to zero, the object is automatically deallocated. As a C extension developer, it's your job to manage these counts precisely.

✅ Actions That Increase Reference Count
Creating new Python objects (e.g., PyLong_FromLong(), PyList_New(), PyUnicode_FromString())
→ Return a new reference (refcount = 1).

Calling Py_INCREF(obj)
→ Explicitly increments the reference count by 1.

Py_BuildValue()
→ Constructs new Python objects and returns them with refcount = 1.

Calling Python functions in C
→ Functions like PyObject_CallObject() return a new reference.

Returning from a C function
→ When returning a Python object to Python, it must be a new reference.

⚠️ Borrowed References (Don’t Increase Count Automatically)
Accessors like PyTuple_GetItem(), PyList_GetItem(), PyDict_GetItemString() return borrowed references.

You must call Py_INCREF() yourself if you plan to store or reuse the object.

## Contiguous

Yes — a NumPy array can be non-contiguous in memory.

This happens when the array:

Is sliced, transposed, or reshaped in certain ways,

Is created from views of other arrays (not copies),

Is Fortran-ordered (column-major) instead of C-ordered (row-major),

Has been broadcasted or strided unusually.