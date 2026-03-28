//
// Created by aben-ham on 10/8/25.
//

#ifndef TYPES_H
#define TYPES_H

#if defined(FLOAT_TYPE) && FLOAT_TYPE == float
    #define NPY_FLOAT_TYPE NPY_FLOAT32
    #define FDX_T float
#else
    #define NPY_FLOAT_TYPE NPY_FLOAT64
    #define FDX_T double
    #define FLOAT_TYPE double
#endif

#if defined(INT_TYPE) && INT_TYPE == int32
    #define NPY_INT_TYPE NPY_INT32
#else
    // default branch for int type
    # define INT_TYPE int64
    # define NPY_INT_TYPE NPY_INT64
#endif

// raise an error if MODULE_NAME macro does nor exist
#ifndef MODULE_NAME
    #error "MODULE_NAME is not defined"
#endif


// Convert macro to string
#define _STR(x) #x
#define STR(x) _STR(x)

typedef FDX_T fdx_t;
//idx_t already defined in metis

#endif //TYPES_H
