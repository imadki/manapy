#include <numpy/arrayobject.h>
#include <array>
#include <sstream>
#include <metis.h>

#ifndef PYARRAY_H
#define PYARRAY_H

/**
 * Helper to build a std::array of dimensions from a variadic list.
 *
 * @brief  Convert a C++ variadic sequence of integer sizes into a
 *         std::array<npy_intp, N> at compile time.
 */
template <class... Ints>
constexpr auto make_npy_dims(Ints... ns)
    -> std::array<npy_intp, sizeof...(Ints)>
{
    return { static_cast<npy_intp>(ns)... };
}

/**
 * Type‑to‑NPY mapping.
 *
 * @brief  Map a C++ fundamental type to the corresponding NumPy type number
 *         (e.g. int32_t → NPY_INT32).  This is used as the compile‑time
 *         constant `valueType` for each specialization.
 */
template <typename Type>
struct PyArrayType {
    static_assert(true, "Type error."); // only instantiate the specialisations
};


/**
 * Specialisations – one per primitive type.
 *
 * @brief  Convert a C++ fundamental type to its NumPy type number.
 */
template <>
struct PyArrayType<int32_t> {
    static constexpr int valueType = NPY_INT32;
};

template <>
struct PyArrayType<int64_t> {
    static constexpr int valueType = NPY_INT64;
};

template <>
struct PyArrayType<int8_t> {
    static constexpr int valueType = NPY_INT8;
};


template <>
struct PyArrayType<float> {
    static constexpr int valueType = NPY_FLOAT32;
};

template <>
struct PyArrayType<double> {
    static constexpr int valueType = NPY_FLOAT64;
};


/**
 * Fixed‑dimensional NumPy wrapper.
 *
 * @brief  A thin C++ façade around a NumPy ``PyArrayObject``.  The class is
 *         templated on the element type (T) and the number of dimensions
 *         (Dim).  All memory ownership stays with the wrapper – it calls
 *         ``Py_XDECREF`` in its destructor.
 *
 * @details The implementation uses only NumPy C‑API functions:
 *          - ``PyArray_SimpleNew`` / ``PyArray_ZEROS`` to allocate buffers.
 *          - ``sub_array`` builds a new wrapper that owns its own slice buffer.
 *
 *         Copy/assignment are disabled because sharing the original object
 *         would lead to double‑free.  The class is deliberately **non‑copyable**
 *         and therefore cannot be used for “shallow” copies of NumPy objects.
 */
template <typename Type, int Dim>
class PyArray {
private:
    char *data;              ///< Pointer to the raw data buffer.
    npy_intp *strides;       ///< Stride between consecutive rows/axes in bytes
    int nd;                  ///< number of dimensions (equal to Dim)
public:
    npy_intp *shape;                ///< Dimensionality of this object
    PyArrayObject *ref_holder;      ///< Reference to free the PyArray object

    // ---------------------------------------------------------------------------
    // Compile‑time constants used for all constructors.
    // ---------------------------------------------------------------------------
    static constexpr int valueType = PyArrayType<Type>::valueType;

public:
    // delete these construct to prevent double free on ref_holder
    PyArray(const PyArray&) = delete;                   // delete copy constructor
    PyArray& operator=(const PyArray&) = delete;        // delete copy assignment
    // no need of default constructor
    PyArray() = delete;

    /**
     *  Allocate a new buffer (filled with zeros or uninitialised).
     *
     * @brief  Create a ``PyArray`` from its dimensions.  If ``init_with_zeros`` is
     *         true the buffer is filled with zeroes; otherwise it contains whatever
     *         the caller writes.
     *
     * @details The function calls NumPy’s ``PyArray_SimpleNew`` (empty) or
     *          ``PyArray_ZEROS`` (zero‑filled).  On failure a ``std::bad_alloc`` is
     *          thrown – the wrapper re‑throws it unchanged so that the allocation
     *          problem propagates to the caller.
    *
    * @param   dims        dimensions of the array, supplied as a std::array<npy_intp,Dim>.
    * @param   init_with_zeros  (optional) whether to zero‑initialise the buffer.
    */
    explicit PyArray(const std::array<npy_intp, Dim> &shape, const bool init_with_zeros = false) {
        PyArrayObject *new_array = nullptr;
        if (init_with_zeros) {
            new_array = (PyArrayObject *)PyArray_ZEROS(Dim, shape.data(), PyArray::valueType, 0);
        } else {
            new_array = (PyArrayObject *)PyArray_SimpleNew(Dim, shape.data(), PyArray::valueType);
        }
        if (!new_array) {
            throw std::bad_alloc();
        }
        this->data = ((PyArrayObject_fields *)new_array)->data;
        this->strides = ((PyArrayObject_fields *)new_array)->strides;
        this->nd = ((PyArrayObject_fields *)new_array)->nd;
        this->shape = ((PyArrayObject_fields *)new_array)->dimensions;
        this->ref_holder = new_array;
    }

    /**
     *  Constructor that accepts an existing ``PyArrayObject``.
     *
     * @brief  Wrap another NumPy array – no copy is made, the wrapper just
     *         stores a pointer to it.  The original object stays alive independently.
     *
     * @details A sanity check guarantees that:
     *          - the number of dimensions matches ``Dim``,
     *          - the element type matches ``valueType``,
     *          - and the array is contiguous (so strides are all 1).
     *         If any condition fails a ``std::runtime_error`` is thrown.
    *
    * @param   arr_obj  The existing NumPy object to wrap.
    */
    explicit PyArray(PyArrayObject *arr_obj):
        data(((PyArrayObject_fields *)arr_obj)->data),
        strides(((PyArrayObject_fields *)arr_obj)->strides),
        nd(((PyArrayObject_fields *)arr_obj)->nd),
        shape(((PyArrayObject_fields *)arr_obj)->dimensions),
        ref_holder(nullptr)
    {

        if (nd != Dim || PyArray_TYPE(arr_obj) != PyArray::valueType || !PyArray_ISCONTIGUOUS(arr_obj)) {
            throw std::runtime_error("Error in PyArray constructor (different Dimension, type or alignement type)");
        }
    }

    /**
     * @brief  Snapshot of a C‑object without owning it – !! used only by sub_array().
     *
     * @details No copy is performed; the caller owns the buffer. This
     *          constructor stores pointers to ``data``, ``strides`` and ``shape``,
     *          while leaving ``ref_holder`` as nullptr because the C‑object is not
     *          owned by this wrapper.
    */
    PyArray(char *data, npy_intp *strides, npy_intp *shape):
        data(data),
        strides(strides),
        nd(Dim),
        shape(shape),
        ref_holder(nullptr)
    {}

    /**
     *  Destructor – releases the owned ``PyArrayObject``.
     *
     * @brief  Releases ownership of the NumPy object stored in ``ref_holder``.
     */
    ~PyArray() {
        Py_XDECREF(this->ref_holder); // ``X_`` means ignore null pointer
        this->ref_holder = nullptr;
    }


    /** -----------------------------------------------------------------------
     *  Sub‑array builder – removes the first dimension and creates a new wrapper.
     *
     * @brief  Returns a ``PyArray<T,Dim-1>`` that is a view of the original array,
     *         with its leading dimension indexed by ``index``.  The new buffer
     *         owns its own stride array; it is not a copy of the data.
     *
     * @details The operation is performed by pointer arithmetic on the underlying
     *          NumPy buffer, which is safe because the original buffer is guaranteed
     *          to be contiguous (``strides[0] == 1``).  ``Dim-1 > 0`` is a static‑assert.
    *
    * @param   index  Index into the first dimension that should be dropped.
    */
    PyArray<Type, Dim - 1> sub_array(npy_intp index) const noexcept {
        static_assert(Dim - 1 > 0, "Can't construct sub array with zero dimension");

        return PyArray<Type, Dim - 1>(
            this->data + index * this->strides[0],
            this->strides + 1,
            this->shape + 1
            );
    }

    /** @brief Return a reference to the last element.*/
    Type &last() const noexcept {
        return *(reinterpret_cast<Type *>(this->data) + (this->shape[0] - 1));
        //return *(Type *)(this->data + (this->shape[0] - 1) * this->strides[0]);
    }

    /** @brief Return a reference to the last element at raw ``i`` in a two‑dimensional view. */
    Type &last2(const idx_t i) const noexcept {
        return *(reinterpret_cast<Type *>(this->data) + (i + 1) * this->shape[1] - 1);
        //return *(Type *)(this->data + i * this->strides[0] + (this->shape[1] - 1) * this->strides[1]);
    }

    /** @brief Return a reference to the element at position ``i``. */
    Type &get(const idx_t i) const noexcept {
        return *(reinterpret_cast<Type *>(this->data) + i);
        // return *(Type *)(this->data + i * this->strides[0]);
    }

    /**
     * @brief  Return a reference to the element at position ``(i,j)`` in a two‑dimensional view.
    */
    Type &get2(const idx_t i, const idx_t j) const noexcept {
        return *(reinterpret_cast<Type *>(this->data) + i * this->shape[1] + j);
        //return *(Type *)(this->data + i * this->strides[0] + j * this->strides[1]);
    }

};

#endif //PYARRAY_H
