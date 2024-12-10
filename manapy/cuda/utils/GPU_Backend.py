from numba import cuda
import inspect
import numpy as np
from numba import cuda
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class GPU_Backend():
    float_precision = 'float32'
    int_precision = 'int32'
    cache = True
    nb_blocks = 32
    nb_threads = 32
    free = False
    stream = None

    @staticmethod
    def init_stream():
        GPU_Backend.stream = None
        
        #### Select Device By Rank
        gpu_id = RANK % len(cuda.gpus)
        cuda.select_device(gpu_id)
        gpu_name = cuda.gpus[0].name
        print(f"I am the rank {RANK} i'm using this GPU {gpu_name} with id {gpu_id}")
        GPU_Backend.stream = cuda.default_stream()


    @staticmethod
    def set_config(float_precision, int_precision, cache):
        GPU_Backend.float_precision = float_precision
        GPU_Backend.int_precision = int_precision
        GPU_Backend.cache = cache

    @staticmethod
    def get_arg_types(func, float_precision, int_precision):
        # Get the function's argument types
        arg_types = []
        sig = inspect.signature(func)
        arg_names = sig.parameters.keys()
        for arg_name in arg_names:
            arg_type = sig.parameters[arg_name].annotation
            arg_type = arg_type.replace("float", float_precision)
            arg_type = arg_type.replace("uint32", int_precision)
            arg_types.append(arg_type)
        return_type = sig.return_annotation
        if return_type == inspect._empty:
            return_type = 'void'
        return str(return_type), tuple(arg_types)

    @staticmethod
    def compile_kernel(fun, device = False):
        return_type, signature = GPU_Backend.get_arg_types(fun, GPU_Backend.float_precision, GPU_Backend.int_precision)
        signature = '(' + ', '.join(signature) + ')'
        signature = f'{return_type}{signature}'
  
        #print(f"compile {fun.__name__} to cuda => signature={signature}")
        print(f"compile {fun.__name__} to cuda")

        if device == False:
            new_fun = cuda.jit(signature, fastmath=True, cache=GPU_Backend.cache, device=device)(fun)
        else:
            new_fun = cuda.jit(signature, fastmath=True, cache=GPU_Backend.cache, device=device)(fun)

        return new_fun
    
    @staticmethod
    def get_gpu_prams(size):
      if GPU_Backend.free == True:
        return (size // GPU_Backend.nb_threads + 1, GPU_Backend.nb_threads)
      return (GPU_Backend.nb_blocks, GPU_Backend.nb_threads)

