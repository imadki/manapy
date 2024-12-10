import pycuda.autoinit
import pycuda.driver as drv
import time
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


#load kernel code
mod = SourceModule(open("./MatrixMulKernel.cu", "r").read())
multiply_them = mod.get_function("matrix_mul")

#init
M = 1024
K = 1024
N = 1024
a = np.random.rand(M * K).astype(np.float32)
b = np.ones(K * N).astype(np.float32)
c = np.zeros(M * N).astype(np.float32)


#cuda allocation
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

#copy to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

#exec
number_of_threads = (32, 32, 1)
number_of_grid = (256, 256)
multiply_them(a_gpu, b_gpu, c_gpu, 
    np.int32(M), np.int32(K), np.int32(N), block=number_of_threads, grid=number_of_grid)

#copy to host
cuda.memcpy_dtoh(c, c_gpu)

#check
a = a.reshape((M, K))
b = b.reshape((K, N))
c = c.reshape((M, N))

cpu_res = np.dot(a, b)
np.testing.assert_almost_equal(c, cpu_res, decimal=2, )

#performance
start_time = time.time()

nb_time = 2
for i in range(nb_time):
  multiply_them(a_gpu, b_gpu, c_gpu, 
    np.int32(M), np.int32(K), np.int32(N), block=number_of_threads, grid=number_of_grid)
  cuda.Context.synchronize()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time = elapsed_time / nb_time

print(elapsed_time)
