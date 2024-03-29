# -*- coding: utf-8 -*-
from manapy.backends.base import Backend
from manapy.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d
from manapy.backends.cpu.local import stack_empty_impl
from pyccel.epyccel import epyccel, get_unique_name
from pyccel.codegen.pipeline import execute_pyccel
from importlib.machinery import ExtensionFileLoader
import sys
import importlib


import pythran
import inspect
import hashlib
import imp
import re

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

class Pythranjit(object):

    def __init__(self, cache=False, float_precision=None, int_precision=None, **flags):
        self.flags = flags
        self.cache = cache
        self.float_precision = float_precision
        self.int_precision   = int_precision
        
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
        
        header = "#pythran export {}({})\n".format(fname, ", ".join(get_arg_types(fun, self.float_precision, self.int_precision)))
        header += "import numpy as np \n"
        
        output_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(module_dir , '.')
        
        #FIXME: implement a cache here
        module_name = "pythranized_" + fname
        self.flags["extra_compile_args"] = ["-Os", "-march=native", "-lopenblas"]
    
        #FIXME: force output in tmp dir
        module_path = pythran.compile_pythrancode(module_name, header + src, **self.flags)
        output_path = os.path.join(output_dir, module_name + ".cpython-38-x86_64-linux-gnu.so")
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
                print("here")
                return cuda.jit(nopython=True, fastmath=True, device=True, signature_or_function=signature, cache=self.cache)(func)[2,2]
             
            else:
                if (self.multithread == 'single'):
                    print(signature)
                    return nb.jit(nopython=True, fastmath=True, signature_or_function=signature, cache=self.cache)(func)
                else:
                    return nb.jit(nopython=True, fastmath=True, parallel=True, signature_or_function=signature, cache=self.cache)(func)
        
        elif backend == "pythran":
            J = Pythranjit(cache=self.cache, float_precision=self.float_precision, int_precision=self.int_precision)
            return J(func)
        
        elif backend == "pyccel":
            
            arg_types = get_arg_types(func, self.float_precision, self.int_precision)

            new_params = [inspect.Parameter(name, inspect._ParameterKind.POSITIONAL_OR_KEYWORD, annotation=type) \
                          for name, type in zip(inspect.signature(func).parameters.keys(), arg_types)]
            new_sig = inspect.signature(func).replace(parameters=new_params)
            func.__signature__ = new_sig
            
            # Get the source lines of the function body
            source_lines = inspect.getsource(func)
            lines = source_lines.split('\n')
            # Find the index of the line containing the colon
            index = next(i for i, line in enumerate(lines) if '):' in line)
    
            # Remove all lines before and including the line with the colon
            source_lines = lines[index+1:]
            
            source_lines = ["    import numpy as np"] + source_lines
            # Construct the new function definition line based on the new signature
            new_def_line = "def {}{}:".format(func.__name__, new_sig)
            
            # Combine the new definition line with the original source lines
            new_source = [new_def_line] + source_lines
            new_source = '\n'.join(new_source)
            
            module = inspect.getmodule(func)
            # Get the path of the module
            module_path = inspect.getfile(module)
            # Get the directory path of the module
            module_dir = os.path.dirname(module_path)
            
            language = "fortran"
             # Define working directory 'folder'
            folder = module_dir
            epyccel_dirname = '__epyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
            epyccel_dirpath = os.path.join(folder, epyccel_dirname)
            
            print(epyccel_dirpath)
            
            module_name, module_lock = get_unique_name('mod', epyccel_dirpath)
            pymod_filename = '{}.py'.format(module_name)
           
            with open(pymod_filename, 'w') as f:
                 f.writelines(new_source)
            execute_pyccel(pymod_filename,
                           language  = language)            
            
            # Import shared library
            sys.path.insert(0, epyccel_dirpath)
#            os.remove(pymod_filename)

            # http://ballingt.com/import-invalidate-caches
            # https://docs.python.org/3/library/importlib.html#importlib.invalidate_caches
            importlib.invalidate_caches()
    
            package = importlib.import_module(module_name)
            sys.path.remove(epyccel_dirpath)
    
            if language != 'python':
                # Verify that we have imported the shared library, not the Python one
                loader = getattr(package, '__loader__', None)
                if not isinstance(loader, ExtensionFileLoader):
                    raise ImportError('Could not load shared library')
    
            # If Python object was function, extract it from module
            func = getattr(package, func.__name__)
            
            return func

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

