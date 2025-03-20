# -*- coding: utf-8 -*-
from manapy.backends.base import Backend
from manapy.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d
from manapy.backends.cpu.local import stack_empty_impl
from importlib.machinery import ExtensionFileLoader
import sys
import importlib


import inspect

from numba.extending import register_jitable

import numba as nb
from numba import cuda
import numpy as np
import os

def get_arg_types(func, float_precision, int_precision):
    # Get the function's argument types
    arg_types = []
    arg_names = inspect.signature(func).parameters.keys()
    for arg_name in arg_names:
        arg_type = inspect.signature(func).parameters[arg_name].annotation
        arg_type = arg_type.replace("float", float_precision)
        arg_type = arg_type.replace("uint32", int_precision)
        arg_types.append(arg_type)
    return tuple(arg_types)

class CPUBackend(Backend):
    """
    Backend for CPU computation
    - Support single thread and multi threads
    - Just-in Time compile via Numba
    """
    name = 'cpu'

    def __init__(self, multithread="parallel", backend=False, cache=False, float_precision=None, int_precision=None):
        # Get mutli-thread type
        
        self.backend     = backend
        self.multithread = multithread
        self.cache       = cache
        self.device = "cpu"
        
        if float_precision == "single":
            self.float_precision   = "float32"
        else:
            self.float_precision   = "float64"
        
        if int_precision == "signed":
            self.int_precision   = "int32"
        else:
            self.int_precision   = "uint32"
        
        
        # Loop structure for multi-thread type
        if multithread == 'single':
            self.make_loop = make_serial_loop1d
            # Enforce to disable OpenMP
            os.environ['OMP_NUM_THREADS'] = '1'
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if multithread in ['default', 'forksafe', 'threadsafe', 'safe', 'omp', 'tbb']:
                nb.config.THREADING_LAYER = multithread
    
    def compile(self, func, signature=False, forcedbackend=None, outer=False):
        
        backend = self.backend
        if forcedbackend is not None:
            backend = forcedbackend
        
        if backend=="numba":
            if signature:
                signature = get_arg_types(func, self.float_precision, self.int_precision)
                signature = '(' + ', '.join(signature) + ')'
                signature = str(signature)
            else:
                signature = None
            
            if self.device == "gpu":
                return cuda.jit(nopython=True, fastmath=True, device=True, signature_or_function=signature, cache=self.cache)(func)[2,2]
             
            else:
                if (self.multithread == 'single'):
                    return nb.jit(nopython=True, fastmath=True, signature_or_function=signature, cache=self.cache)(func)
                else:
                    return nb.jit(nopython=True, fastmath=True, parallel=True, signature_or_function=signature, cache=self.cache)(func)
        
        else:
            return func
            
    def local_array(self):
        # Stack-allocated array
        # Original code from https://github.com/numba/numba/issues/5084
        # Modified for only 1-D array
        
        np_dtype = np.float64

        @register_jitable(inline='always')
        def stack_empty(shape, dtype=np_dtype):
            arr_ptr=stack_empty_impl(shape[0], dtype)
            arr=nb.carray(arr_ptr, shape)
            return arr

        return stack_empty

