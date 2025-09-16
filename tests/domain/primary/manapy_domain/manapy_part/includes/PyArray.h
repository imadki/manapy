#include "manapy_part.h"
#include <sstream>
#include <array>

#ifndef PYARRAY_H
#define PYARRAY_H

template <class... Ints>
constexpr auto make_npy_dims(Ints... ns)
    -> std::array<npy_intp, sizeof...(Ints)>
{
    return { static_cast<npy_intp>(ns)... };
}

template <typename Type>
struct PyArrayType {
    static_assert(true, "Type error.");
};

template <>
struct PyArrayType<int32_t> {
    static constexpr int valueType = NPY_INT32;
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



template <typename Type, int Dim>
class PyArray {
public:
    char *data;
    npy_intp *strides;
    npy_intp *shape;
    int nd;
    PyArrayObject *ref_holder; // reference to free the object

    static constexpr int valueType = PyArrayType<Type>::valueType;

public:

    PyArray() {
        this->data = nullptr;
        this->strides = nullptr;
        this->shape = nullptr;
        this->nd = 0;
        this->ref_holder = nullptr;
    }

    PyArray(const std::array<npy_intp, Dim> &shape) {
        auto *new_array = (PyArrayObject *)PyArray_SimpleNew(Dim, shape.data(), PyArray::valueType);
        if (!new_array) {
            throw std::bad_alloc();
        }
        this->data = ((PyArrayObject_fields *)new_array)->data;
        this->strides = ((PyArrayObject_fields *)new_array)->strides;
        this->shape = ((PyArrayObject_fields *)new_array)->dimensions;
        this->nd = ((PyArrayObject_fields *)new_array)->nd;
        this->ref_holder = new_array;
    }


    explicit PyArray(PyArrayObject *arr_obj) {
        if (!arr_obj)
            throw std::runtime_error("Error in PyArray constructor");
        this->data = ((PyArrayObject_fields *)arr_obj)->data;
        this->strides = ((PyArrayObject_fields *)arr_obj)->strides;
        this->shape = ((PyArrayObject_fields *)arr_obj)->dimensions;
        this->nd = ((PyArrayObject_fields *)arr_obj)->nd;
        this->ref_holder = arr_obj;

        if (nd != Dim || PyArray_TYPE(arr_obj) != PyArray::valueType || !PyArray_ISCONTIGUOUS(arr_obj)) {
            throw std::runtime_error("Error in PyArray constructor");
        }
    }

    // PyArray &operator=(PyArray &&other) noexcept {
    //     PyArrayObject *tmp = other.ref_holder;
    //
    //     other.ref_holder = nullptr;
    //
    //     this->data = other.data;
    //     this->strides = other.strides;
    //     this->shape = other.shape;
    //     this->nd = other.nd;
    //     this->ref_holder = tmp;
    //
    //
    //
    //     return *this;
    // }

    ~PyArray() {
        Py_XDECREF(this->ref_holder);
        this->ref_holder = nullptr;
    }



    PyArray<Type, Dim - 1> sub_array(npy_intp index) const {
        PyArray<Type, Dim - 1> ret;

        if (this->nd <= 1)
            throw std::runtime_error("Cannot access non-array with shape 0");
        ret.data = this->data + index * this->strides[0];
        ret.strides = this->strides + 1;
        ret.shape = this->shape + 1;
        ret.nd -= 1;
        ret.ref_holder = nullptr;

        return ret;
    }

    Type &last() const {
        return *(Type *)(this->data + (this->shape[0] - 1) * this->strides[0]);
    }

    Type &get(const idx_t index) const {
        return *(Type *)(this->data + index * this->strides[0]);
    }



    // void    describe() const {
    //     std::ostringstream os;
    //
    //
    //     os << "first_ele=" << *(const Type *)this->data << " ";
    //
    //     os << "nd=" << this->nd << " ";
    //
    //     os << "shape=[";
    //     for (int i = 0; i < this->nd; ++i) {
    //         if (i) os << ",";
    //         os << this->shape[i];
    //     }
    //     os << "] ";
    //
    //     os << "strides=[";
    //     for (int i = 0; i < this->nd; ++i) {
    //         if (i) os << ",";
    //         os << this->strides[i];
    //     }
    //     os << "]\n";
    //
    //     //print_instant("%s", os.str().c_str());
    // }

    // void    print() const {
    //     std::ostringstream os;
    //     this->_print_recursive(os, *this);
    //     //print_instant("%s\n", os.str().c_str());
    // }

private:


    // void _print_recursive(std::ostringstream &os, const PyArray<Type> &obj) const {
    //     constexpr int limit = 100;
    //
    //     if (obj.nd == 1) {
    //         os << "[";
    //         for (int i = 0; i < obj.shape[0]; i++) {
    //             if (i) os << ",";
    //             os << obj.get(i);
    //             if (i == limit && limit != obj.shape[0] - 1) {
    //                 os << "... " << obj.last();
    //                 break;
    //             }
    //         }
    //         os << "]";
    //     } else {
    //         os << "[";
    //         for (int i = 0; i < obj.shape[0]; i++) {
    //             if (i) os << ",";
    //             this->_print_recursive(os, obj.sub_array(i));
    //             if (i == limit && limit != obj.shape[0] - 1) {
    //                 os << "... ";
    //                 this->_print_recursive(os, obj.sub_array(obj.shape[0] - 1));
    //                 break;
    //             }
    //         }
    //         os << "]";
    //     }
    // }
};

#endif //PYARRAY_H
