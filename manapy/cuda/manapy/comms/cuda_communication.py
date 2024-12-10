import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)


# ✅ ❌ 🔨
# define_halosend ✅

def get_kernel_define_halosend():
  def kernel_define_halosend(
    w_c:'float[:]', 
    w_halosend:'float[:]', 
    indsend:'int32[:]'
    ):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, w_halosend.shape[0], stride):
      w_halosend[i] = 0
      if i < indsend.shape[0]:
        w_halosend[i] = w_c[indsend[i]]

  kernel_define_halosend = GPU_Backend.compile_kernel(kernel_define_halosend)

  def result(*args):
    # w_halosend
    VarClass.debug(kernel_define_halosend, args)
    args = [VarClass.to_device(arg) for arg in args]
    size = len(args[1])
    nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
    kernel_define_halosend[nb_blocks, nb_threads, GPU_Backend.stream](*args)
    GPU_Backend.stream.synchronize()

  return result  