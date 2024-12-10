from mpi4py import MPI
import numba
from numba import cuda
#import cupy as cp
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(f'Runk : {rank} Size : {size}')

if rank == 0:
  print("Run 0 is talking")