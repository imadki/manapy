# -*- coding: utf-8 -*-
import numba as nb
import math


def make_parallel_loop1d(ne, func, n0=0):
    # Get number of threads
    n = ne - n0
    num_threads = nb.get_num_threads()
    num_per_thread = int(math.ceil(n / num_threads))

    # Compile inner function
    _func = nb.jit(nopython=True, fastmath=True)(func)
    
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def loop(*args):        
        # Split loop by number of threads
        for index_thread in nb.prange(num_threads):
            i_begin = n0 + index_thread * num_per_thread
            i_end = n0 + min(n, (index_thread + 1) * num_per_thread)
            
            # Run inner function
            _func(i_begin, i_end, *args)               
                
    return loop
            
            
def make_serial_loop1d(ne, func, n0=0, debug=False):
    # Compile function
    if debug:
        # Don't JIT compile if debug mode
        return lambda *args : func(n0, ne, *args)
    else:
        # Compile inner function
        _func = nb.jit(nopython=True, fastmath=True)(func)

        # Generate serial loop
        def loop(*args):
            _func(n0, ne, *args)

        # Compile whole loop
        return nb.jit(nopython=True, fastmath=True)(loop)
