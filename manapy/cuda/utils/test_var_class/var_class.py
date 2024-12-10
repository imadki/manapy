from manapy.cuda.utils import VarClass
import numpy as np
from numba import cuda
import numba


original_array = np.array
original_zeros = np.zeros
original_ones = np.ones
original_empty = np.empty
original_full = np.full

def array(*args, **kwargs):
    return np.asarray(original_array(*args, **kwargs)).view(VarClass)

def zeros(shape, dtype=float, order='C'):
    return np.asarray(original_zeros(shape, dtype, order)).view(VarClass)

def ones(shape, dtype=float, order='C'):
    return np.asarray(original_ones(shape, dtype, order)).view(VarClass)

def empty(shape, dtype=float, order='C'):
    return np.asarray(original_empty(shape, dtype, order)).view(VarClass)

def full(shape, fill_value, dtype=None, order='C'):
    return np.asarray(original_full(shape, fill_value, dtype, order)).view(VarClass)

# Patch the numpy functions
print("Patch the numpy functions np.array, np.zeros, np.ones, np.empty, np.full")
np.array = array
np.zeros = zeros
np.ones = ones
np.empty = empty
np.full = full


r = np.array([8, 9], dtype='float32')
r = np.zeros(20)

print(type(r))
