import numpy as np

FLOAT_TYPE = "float32"
np_float_type = np.float32
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

