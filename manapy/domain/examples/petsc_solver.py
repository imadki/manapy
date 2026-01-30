import sys
import petsc4py
petsc4py.init(sys.argv)
import scipy
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

def to_dat(petscA):
  # A = scipy.io.mmread("A.mtx").tocsr()
  #
  # petscA = PETSc.Mat().createAIJ(
  #   size=A.shape,
  #   csr=(A.indptr, A.indices, A.data)
  # )

  viewer = PETSc.Viewer().createBinary("A.dat", "w")
  petscA.view(viewer)

def read_mat_fast(filename):
  viewer = PETSc.Viewer().createBinary("A.dat", "r")
  A = PETSc.Mat().load(viewer)
  return A

  A_scipy = scipy.io.mmread(filename).tocsr()
  M, N = A_scipy.shape

  A = PETSc.Mat().createAIJ(
    size=(M, N),
    csr=(A_scipy.indptr, A_scipy.indices, A_scipy.data),
    comm=PETSc.COMM_WORLD
  )
  to_dat(A)
  return A




def read_vec(filename):
  b_scipy = scipy.io.mmread(filename).flatten()
  M = len(b_scipy)

  b = PETSc.Vec().create(comm=comm)
  b.setSizes(M)
  b.setFromOptions()

  rstart, rend = b.getOwnershipRange()
  b.setValues(range(rstart, rend), b_scipy[rstart:rend])
  b.assemble()
  to_dat(b)
  return b

# -----------------------------
# MPI communicator
# -----------------------------
comm = PETSc.COMM_WORLD
rank = comm.getRank()

# -----------------------------
# Load matrices
# -----------------------------
print("Reading A")
A = read_mat_fast("A.mtx")
print("Reading b")
b = read_vec("b.mtx")
print("Create x")
x = b.duplicate()
x.zeroEntries()

# -----------------------------
# Create linear solver KSP
# -----------------------------
print("Start")
ts = MPI.Wtime()



ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setTolerances(
    rtol=1e-8,
    atol=1e-12,
    divtol=1e5,
    max_it=10000
)

# -----------------------------------
# General non-symmetric
"""
gmres
fgmres
lgmres
dgmres
bcgs
bicg
ibcgs
tfqmr
qcg
"""
# Symmetric / SPD
"""
cg
cr
pipecg
pipecr
"""
# -----------------------------------
ksp.setType("gmres")

pc = ksp.getPC()
pc.setType("gamg")   # try "jacobi", "hypre", "lu", gamg

ksp.setFromOptions()
ksp.solve(b, x)

te = MPI.Wtime()
tt = MPI.COMM_WORLD.reduce(te - ts, op=MPI.MAX, root=0)
# -----------------------------
# Output info
# -----------------------------
if rank == 0:
    print("Converged in", ksp.getIterationNumber(), "iterations")
    print("Final residual norm:", ksp.getResidualNorm())
    print("End")
    print("Time to do calculation", tt)

# -----------------------------
# Save solution x
# -----------------------------
x_local = x.getArray()
sizes = MPI.COMM_WORLD.allgather(len(x_local))

if MPI.COMM_WORLD.Get_rank() == 0:
    x_global = np.empty(sum(sizes))
else:
    x_global = None

MPI.COMM_WORLD.Gatherv(
    x_local,
    (x_global, sizes) if MPI.COMM_WORLD.Get_rank() == 0 else None,
    root=0
)

if MPI.COMM_WORLD.Get_rank() == 0:
    scipy.io.mmwrite("sol_x.mtx", x_global.reshape(-1, 1))

# viewer = PETSc.Viewer().createASCII("sol_x.mtx", "w")
# viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATRIXMARKET)
# x.view(viewer)
