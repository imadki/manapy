import numpy as np
from numba import cuda
from manapy.cuda.utils import (
    VarClass,
    GPU_Backend
)


# ✅ ❌ 🔨
# get_kernel_convert_solution 🔨 #! need test
# get_kernel_facetocell 🔨 #! need test
# get_kernel_celltoface ✅


def get_kernel_convert_solution():
    

    def kernel_convert_solution(x1:'float[:]', x1converted:'float[:]', tc:'int32[:]', b0Size:'int32'):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)

        for i in range(start, x1converted.shape[0], stride):
            x1converted[i] = x1[tc[i]]

    kernel_convert_solution = GPU_Backend.compile_kernel(kernel_convert_solution)

    def result(*args):
        VarClass.debug(kernel_convert_solution, args)
        args = [VarClass.to_device(arg) for arg in args]
        size = len(args[1]) #x1converted
        nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
        kernel_convert_solution[nb_blocks, nb_threads, GPU_Backend.stream](*args)
        GPU_Backend.stream.synchronize()

    return result

def get_kernel_facetocell():
    def kernel_facetocell(
        u_face:'float[:]',
        u_c:'float[:]',
        faceidc:'int32[:,:]',
        dim:'int32'
        ):

        start = cuda.grid(1)
        stride = cuda.gridsize(1)
    
        #? u_c[:] = 0.
        for i in range(start, u_c.shape[0], stride):
            u_c[i] = 0.
            for j in range(faceidc[i][-1]):
                u_c[i] += u_face[faceidc[i][j]]
            u_c[i]  /= faceidc[i][-1]

    kernel_facetocell = GPU_Backend.compile_kernel(kernel_facetocell)

    def result(*args):
        VarClass.debug(kernel_facetocell, args)
        args = [VarClass.to_device(arg) for arg in args]
        size = len(args[1]) #u_c
        nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
        kernel_facetocell[nb_blocks, nb_threads, GPU_Backend.stream](*args)
        GPU_Backend.stream.synchronize()

    return result

def get_kernel_celltoface():
    def kernel_celltoface(
        u_cell:'float[:]', 
        u_face:'float[:]', 
        u_ghost:'float[:]', 
        u_halo:'float[:]',
        cellid:'int32[:,:]', 
        halofid:'int32[:]',
        innerfaces:'int32[:]', 
        boundaryfaces:'int32[:]', 
        halofaces:'int32[:]'
        ):
        
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
    
        #? u_c[:] = 0.
        for i in range(start, innerfaces.shape[0], stride):
            c1 = cellid[i][0]
            c2 = cellid[i][1]
            u_face[i] = .5*(u_cell[c1] + u_cell[c2])
            
        for i in range(start, halofaces.shape[0], stride):
            c1 = cellid[i][0]
            u_face[i] = .5*(u_cell[c1] + u_halo[halofid[i]])

        for i in range(start, boundaryfaces.shape[0], stride):
            c1 = cellid[i][0]
            u_face[i] = .5*(u_cell[c1] + u_ghost[i])

    kernel_celltoface = GPU_Backend.compile_kernel(kernel_celltoface)


    def result(*args):
        VarClass.debug(kernel_celltoface, args)
        args = [VarClass.to_device(arg) for arg in args]
        size = max(len(args[6]), len(args[7]), len(args[8])) #innerfaces halofaces boundaryfaces
        nb_blocks, nb_threads = GPU_Backend.get_gpu_prams(size)
        kernel_celltoface[nb_blocks, nb_threads, GPU_Backend.stream](*args)
        GPU_Backend.stream.synchronize()

    return result
            
