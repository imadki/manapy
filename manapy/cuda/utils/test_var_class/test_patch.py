from manapy.cuda.utils import VarClass
from manapy.cuda.utils.test_var_class import patch_numpy_functions
import numpy as np

patch_numpy_functions.patch()

def printInfo(o):
  print("\ttype: ", type(o))
  print("\tdevice: ", o.__on_device__)
  print("\tcuda: ", o.__cuda__)
  print("\ttype_shape: ", o.type_shape)
  print()

z = np.array(5) * -1
z.to_device(z)
printInfo(z)
z.to_device(z)

printInfo(z)

q = z.astype('float32')
printInfo(q)


# #r = np.array(5.0)
# z.to_device()
# z.type_shape = "fff"
# q = z * -1.0
#z = VarClass(z * -1)


# print(q.__on_device__)
# print(q.__cuda__)
# print(z.__on_device__)
# print(z.__cuda__)
#print(q)
# print(z)
#print(q.type_shape)
# print(z.type_shape)

# print(z.type_shape)
# print(type(z))
# print(z)
