//
// Created by aben-ham on 10/8/25.
//

#ifndef TYPES_H
#define TYPES_H

#ifndef MODULE_NAME
# define MODULE_NAME "manapy_part64"
#endif

// Convert macro to string
#define _STR(x) #x
#define STR(x) _STR(x)

#ifndef FLOAT_TYPE
  #define FLOAT_TYPE NPY_FLOAT64
#endif

#ifndef FDX_T
  #define FDX_T float
#endif

typedef FDX_T fdx_t;
const int float_type = FLOAT_TYPE;
// const int int_type = NPY_INT32;

#endif //TYPES_H
