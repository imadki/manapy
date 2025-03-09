import numpy as np
import scipy.sparse as sp
import timeit

from scipy.sparse.linalg import gmres
from scipy.io import mmread
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress the SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

import ls_mumps as mumps

mumps_ls = mumps.MumpsSolver(system="real")
if not mumps.use_mpi:
    raise AttributeError('No mpi4py found! Required by MUMPS solver.')

import sys
filename = "Laplacian1024.mtx"

def OptimAlg(A11, A12, A21, A44, A34, A43):
    """
    print(A43.toarray())                                                                                                                                                                                   
    print(A44.toarray())
    print(A34.toarray())   
    sys.exit()
    """

    A44_inv = sp.linalg.inv(A44)
    D1 = A34 @ A44_inv @ A43
    A11_inv = sp.linalg.inv(A11)
    D2 = A21 @ A11_inv @ A12

    #m, n = A43.shape
    #for i in range(n):
    #    print("sol", (A44_inv @ A43).toarray()[i])#D1.toarray())
    #print(A34)
    #sys.exit()
    #print(D2.toarray())
    #sys.exit()

    return D1, D2

def ComputeM_Additive(R1, R2, R1t, R2t, A1, A2, A, D1, D2):
    
    k1 = D1.shape[1]
    k2 = D2.shape[1]
    A1[-k1:, -k1:] -= D1
    A2[:k2, :k2] -= D2

    AR1 = spsolve(A1, R1)
    AR2 = spsolve(A2, R2)
    
    M = R1t.T @ AR1 + R2t.T @ AR2
#    
    mumps_ls = mumps.MumpsSolver(system="real")
    A1 = coo_matrix(A1)
    mumps_ls.set_mtx_centralized(A1)
    mumps_ls.struct.n = A1.shape[0]
    ARR1 = R1.copy().astype(np.double)
    mumps_ls.set_rhs(ARR1)
    mumps_ls._mumps_call(job=6)
    mumps_ls.__del__()
    
    assert np.allclose(ARR1, AR1, rtol=1e-14, atol=1e-14)
 
#    row, col = A2.nonzero()
#    data = A2[(row, col)]
#    
#    mumps_ls = mumps.MumpsSolver(system="real")
#    mumps_ls.set_rcd_centralized(row+1, col+1, data, A2.shape[0])
#    
#    mumps_ls.set_icntl(18,3)
#    mumps_ls.struct.n = A2.shape[0]
#    ARR2 = R1.copy().astype(np.double)
#    mumps_ls.set_icntl(24, 1)
#    mumps_ls.set_icntl(14, 50)
#    
#    mumps_ls._mumps_call(job=1)
#    #Factorization Phase
#    mumps_ls._mumps_call(job=2)
#    
#    mumps_ls.set_rhs(ARR2)
#    mumps_ls._mumps_call(job=3)
#    #ctx.__del__()
#    print(ARR2)
#
    mumps_ls = mumps.MumpsSolver(system="real")
    A2 = coo_matrix(A2)
    mumps_ls.set_mtx_centralized(A2)
    mumps_ls.struct.n = A2.shape[0]
    ARR2 = R2.copy().astype(np.double)
    mumps_ls.set_rhs(ARR2)
    mumps_ls._mumps_call(job=6)
    mumps_ls.__del__()

    print("ARR2", ARR2)
    assert np.allclose(ARR2, AR2, rtol=1e-14, atol=1e-14)
    

    return M

def solve_linear_system(A, b, M, tol=1e-6, max_iter=100):
    
    x = np.zeros_like(b)
    r = b - A @ x
    
    import timeit
    ts = timeit.default_timer()

    num_iter = 0
    residual_norm = np.linalg.norm(r)
    while residual_norm > tol and num_iter < max_iter:
        z, _ = gmres(A, r, x0=np.zeros_like(b), M=M, restart=2)
        x += z
        r = b - A @ x
        residual_norm = np.linalg.norm(r)
        num_iter += 1
        print("residual norm", residual_norm)
    
    te = timeit.default_timer()
    
    print("cpu time for solving ls using gmres is:", te - ts)
    
    return x, num_iter


def Processing(A, k):
   
    m = A.shape[0]
    n = round(m / 2)
    
    R1 = np.hstack((np.eye(n + k[0]), np.zeros((n + k[0], m - n - k[0]))))
    R2 = np.hstack((np.zeros((m - n + k[1], n - k[1])), np.eye(m - n + k[1])))
    
#    R1 = csr_matrix(R1)
#    R2 = csr_matrix(R2)
    
    ts = timeit.default_timer()
    A1 = R1 @ A @ R1.T
    A2 = R2 @ A @ R2.T
    te = timeit.default_timer()
    print("multiplication using @", te - ts)

    R1t = R1.copy()
    for i in range(n, n + k[0]):
        R1t[i, i] = 0

    R2t = R2.copy()
    for i in range(k[1]):
        R2t[i, n - k[1] + i] = 0    

    A_csc = A.tocsc()
    A11 = A_csc[:n - k[1], :n - k[1]]
    A12 = A_csc[0 : n - k[1], n - k[1] : n]
    A21 = A_csc[n - k[1] : n, 0 : n - k[1]]
    A34 = A_csc[n + 1 : n + k[0] + 1, n + k[0] + 1 : m]
    A44 = A_csc[n + k[0] + 1 : m, n + k[0] + 1 : m]
    A43 = A_csc[n + k[0] + 1 : m, n + 1 : n + k[0] + 1] 

    return A1, A2, R1, R2, R1t, R2t, A11, A12, A21, A44, A34, A43


# Load A from an external matrix file (replace 'matrix.mtx' with your file)
A = mmread(filename).tocsc()


## Load A from an external matrix file (replace 'matrix.mtx' with your file)
#A = mmread('matrix.mtx').tocsc()

# Define the right-hand side b (e.g., a vector of ones)
b = np.ones(A.shape[0])


def Findkkk(A):
#    A = csc_matrix(A)
    m = A.shape[0]
    n = round(m / 2)
    k = np.zeros(4, dtype=np.int32)

    # Finding k(2)
    i = 0
    while A[n:, i].nnz == 0:
        i += 1
    k[1] = n - i

    # Finding k(4)
    i = 0
    while A[n - k[1]:n, i].nnz == 0:
        i += 1
    k[3] = n - k[1] - i

    # Finding k(1)
    i = m - 1
    while A[:n, i].nnz == 0:
        i -= 1
    k[0] = i - n +1

    # Finding k(3)
    i = m - 1
    while A[n:n + k[0], i].nnz == 0:
        i -= 1
    k[2] = i - k[0] - n +1

    return k

# Define k as a list of integers (e.g., [32, 32, 32, 32])
def Findkk(A):
    A = A.toarray()
    m = A.shape[0]
    n = round(m / 2)
    k = np.zeros(4, dtype=int)

    # Finding k(2)
    i = 0
    while np.count_nonzero(A[n:, i]) == 0:
        i += 1
    k[1] = n - i

    # Finding k(4)
    i = 0
    while np.count_nonzero(A[n - k[1]:n, i]) == 0:
        i += 1
    k[3] = n - k[1] - i

    # Finding k(1)
    i = m - 1
    while np.count_nonzero(A[:n, i]) == 0:
        i -= 1
    k[0] = i - n +1

    # Finding k(3)
    i = m - 1
    while np.count_nonzero(A[n:n + k[0], i]) == 0:
        i -= 1
    k[2] = i - k[0] - n +1

    return k

ts = timeit.default_timer()
k = Findkk(A)
te = timeit.default_timer()
print("cpu time for finding k is:", te - ts)

print(k)



ts = timeit.default_timer()
k = Findkkk(A)
te = timeit.default_timer()
print("cpu time for finding k is:", te - ts)

print(k)

# Perform the processing and solve the linear system
ts = timeit.default_timer()
A1, A2, R1, R2, R1t, R2t, A11, A12, A21, A44, A34, A43 = Processing(A, k)
te = timeit.default_timer()
print("cpu time for processing is:", te - ts)

'''
# Perform the processing and solve the linear system
ts = timeit.default_timer()
A1, A2, R1, R2, R1t, R2t, A11, A12, A21, A44, A34, A43 = Processing_cuda(A, k)
te = timeit.default_timer()
print("cpu time for processing cuda is:", te - ts)
'''

ts = timeit.default_timer()
D1, D2 = OptimAlg(A11, A12, A21, A44, A34, A43)
te = timeit.default_timer()
print("cpu time for OptimAlg is:", te - ts)


# Optimal Schwarz algorithm
ts = timeit.default_timer()
M = ComputeM_Additive(R1, R2, R1t, R2t, A1, A2, A, D1, D2)
te = timeit.default_timer()
print("cpu time for compute additive is:", te - ts)

x, num_it = solve_linear_system(A, b, M, tol=1e-6, max_iter=100)

'''
print("Solution:")
for i in range(len(x)):
    print(i, x[i])
print("Number of iterations:", num_it)
'''
'''
# Classical Schwarz method
ts = timeit.default_timer()
M = ComputeM_Schwarz(R1, R2, R1t, R2t, A1, A2, A)
te = timeit.default_timer()
print("cpu time for compute Schwarz is:", te - ts)


x, num_it = solve_linear_system(A, b, M, tol=1e-6, max_iter=100)
#print("Solution:", x)
print("Number of iterations:", num_it)
'''