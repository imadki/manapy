# -*- coding: utf-8 -*-
from manapy.backends.base import Backend
from manapy.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d
from manapy.backends.cpu.local import stack_empty_impl

import pythran
import inspect
import hashlib
import imp
import re

from numba.extending import register_jitable

import numba as nb
import numpy as np
import os

def get_arg_types(func, precision):
    # Get the function's argument types
    arg_types = []
    arg_names = inspect.signature(func).parameters.keys()
    for arg_name in arg_names:
#        print(arg_name)
        arg_type = inspect.signature(func).parameters[arg_name].annotation
        arg_type = arg_type.replace("float", precision)
        arg_types.append(arg_type)
    return tuple(arg_types)

class Pythranjit(object):

    def __init__(self, cache=False, precision=None, **flags):
        self.flags = flags
        self.cache = cache
        self.precision = precision
        
    def __call__(self, fun):
        # FIXME: gather global dependencies using pythran.analysis.imported_ids
        module = inspect.getmodule(fun)
        # Get the path of the module
        module_path = inspect.getfile(module)
        # Get the directory path of the module
        module_dir = os.path.dirname(module_path)
        
        src = inspect.getsource(fun)
        src = re.sub("@.*?\sdef\s","def ", src)
        
        fname = fun.__name__
        m = hashlib.md5()
        m.update(src.encode())  # Encode the string before hashing
        
        header = "#pythran export {}({})\n".format(fname, ", ".join(get_arg_types(fun, self.precision)))
        header += "import numpy as np \n"
        
        print(header, src)
        output_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(module_dir , '.')
        
        #FIXME: implement a cache here
        module_name = "pythranized_" + fname
        self.flags["extra_compile_args"] = ["-Os", "-march=native", "-lopenblas"]
    
        #FIXME: force output in tmp dir
        module_path = pythran.compile_pythrancode(module_name, header + src, **self.flags)
        output_path = os.path.join(output_dir, module_name + ".cpython-38-x86_64-linux-gnu.so")
#        print(output_path)
        os.rename(module_path, output_path)
        module = imp.load_dynamic(module_name, output_path)

        if not self.cache:
            os.remove(output_path)
        
        return getattr(module, fun.__name__)#(*args, **kwargs)

class CPUBackend(Backend):
    """
    Backend for CPU computation
    - Support single thread and multi threads
    - Just-in Time compile via Numba
    """
    name = 'cpu'

    def __init__(self, multithread="single", backend=False, cache=False, precision=None):
        # Get mutli-thread type
        
        self.backend     = backend
        self.multithread = multithread
        self.cache       = cache
        if precision == "single":
            self.precision   = "float32"
        else:
            self.precision   = "float64"
        
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
                signature = get_arg_types(func, self.precision)
                signature = '(' + ', '.join(signature) + ')'
                signature = str(signature)
            else:
                signature = None
                    
            if (self.multithread == 'single'):# or not outer):
                return nb.jit(nopython=True, fastmath=True, signature_or_function=signature, cache=self.cache)(func)
           
            else:
                return nb.jit(nopython=True, fastmath=True, parallel=True, signature_or_function=signature, cache=self.cache)(func)
        
        elif backend == "pythran":
            print(func)
            J = Pythranjit(cache=self.cache, precision=self.precision)
            return J(func)
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

