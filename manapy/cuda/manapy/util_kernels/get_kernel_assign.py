import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)
from manapy.cuda.manapy.util_kernels.other import (
  kernel_assign,
)

#? manapy/ast/functions2d.py

# ✅ ❌ 🔨
# kernel_assign

def get_kernel_assign():
  

  _kernel_assign = GPU_Backend.compile_kernel(kernel_assign)
  
  def result(*args):
    VarClass.debug(_kernel_assign, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[0]) #arr_out
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    _kernel_assign[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result
  
