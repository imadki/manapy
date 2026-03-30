# -*- coding: utf-8 -*-
import numpy as np

FLOAT_TYPE = "float32"
# Only use int64 for larger domain (to be able to index more than 2 Billion cells)
INT_TYPE = "int32"

if FLOAT_TYPE not in ["float32", "float64"]:
    raise ValueError("Unknown float type")
if INT_TYPE not in ["int32", "int64"]:
    raise ValueError("Unknown int type")

np_float_type = np.float32 if FLOAT_TYPE == "float32" else np.float64
np_int_type = np.int32 if INT_TYPE == "int32" else np.int64


# TODO use it inside mesh class
class MeshCell:
    ALLOWED_2D = ['quad', 'triangle']
    ALLOWED_3D = ['pyramid', 'hexahedron', 'tetra']

    TRIANGLE = 1
    QUAD = 2
    TETRA = 3
    HEXAHEDRON = 4
    PYRAMID = 5

    DIC = {
      "triangle": TRIANGLE,
      "quad": QUAD,
      "tetra": TETRA,
      "hexahedron": HEXAHEDRON,
      "pyramid": PYRAMID,
    }


