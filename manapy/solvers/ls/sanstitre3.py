#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:03:31 2025

@author: kissami
"""
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

'''
a = [[ 2.  3.  4.  0.  0.]
[ 3.  0. -3.  0.  6.]
[ 0. -1.  1.  2.  0.]
[ 0.  0.  2.  0.  0.]
[ 0.  4.  0.  0.  1.]]
'''
import ls_mumps as mumps

mumps_ls = mumps.MumpsSolver(system="real")
if not mumps.use_mpi:
    raise AttributeError('No mpi4py found! Required by MUMPS solver.')

# Problem size
n = 100

# Generate a random sparse matrix in coordinate format (COO)
density = 0.01  # 1% nonzero elements
num_entries = int(density * n * n)  # Number of nonzero entries

# Randomly generate row indices, column indices, and values
np.random.seed(42)  # For reproducibility
irn = np.random.randint(0, n, size=num_entries, dtype=np.int32)
jcn = np.random.randint(0, n, size=num_entries, dtype=np.int32)
a = np.random.randn(num_entries)  # Random values

# Ensure the matrix is not too singular
a += 0.1  # Shift to avoid zero entries

b = np.random.randn(n).astype(np.float64)

# mumps_ls = mumps.MumpsSolver(system="real")
# mumps_ls.set_rcd_centralized(irn, jcn, a, n)