from manapy.cuda.utils import VarClass
import numpy as np


def patch():
  if isinstance(np.array([0]), VarClass):
    return
  original_asarray = np.asarray
  original_array = np.array
  original_zeros = np.zeros
  original_ones = np.ones
  original_empty = np.empty
  original_full = np.full

  def asarray(*args, **kwargs):
    print("Create asarray")
    return (VarClass(original_asarray(*args, **kwargs)))

  def array(*args, **kwargs):
    print("Create array")
    return (VarClass(original_array(*args, **kwargs)))

  def zeros(shape, dtype=float, order='C'):
    print("Create zeros")
    return (VarClass(original_zeros(shape, dtype, order)))

  def ones(shape, dtype=float, order='C'):
    print("Create ones")
    return (VarClass(original_ones(shape, dtype, order)))

  def empty(shape, dtype=float, order='C'):
    print("Create empty")
    return (VarClass(original_empty(shape, dtype, order)))

  def full(shape, fill_value, dtype=None, order='C'):
    print("Create full")
    return (VarClass(original_full(shape, fill_value, dtype, order)))

  # Patch the numpy functions
  print("Patch the numpy functions np.array, np.zeros, np.ones, np.empty, np.full")
  np.asarray = asarray
  np.array = array
  np.zeros = zeros
  np.ones = ones
  np.empty = empty
  np.full = full

