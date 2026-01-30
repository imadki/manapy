import sys
import petsc4py
petsc4py.init(sys.argv)
import scipy
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import os
import sys

comm = PETSc.COMM_WORLD
rank = comm.getRank()

if len(sys.argv) != 2:
  print("Usage: python petsc_solver_example.py data_folder")
  sys.exit(1)

def read_mat_A(folder):
  mtx_file = os.path.join(folder, "A.mtx")
  dat_file = os.path.join(folder, "A.dat")
  if os.path.exists(dat_file):
    viewer = PETSc.Viewer().createBinary(dat_file, "r")
    A = PETSc.Mat().load(viewer)
    return A
  elif os.path.exists(mtx_file):
    A_scipy = scipy.io.mmread(mtx_file).tocsr()
    M, N = A_scipy.shape

    A = PETSc.Mat().createAIJ(
      size=(M, N),
      csr=(A_scipy.indptr, A_scipy.indices, A_scipy.data),
      comm=PETSc.COMM_WORLD
    )

    # Save mtx
    viewer = PETSc.Viewer().createBinary(dat_file, "w")
    A.view(viewer)
    return A
  raise RuntimeError("Invalid mat file")


def read_vec_b(folder):
  mtx_file = os.path.join(folder, "b.mtx")
  b_scipy = scipy.io.mmread(mtx_file).flatten()
  M = len(b_scipy)

  b = PETSc.Vec().create(comm=comm)
  b.setSizes(M)
  b.setFromOptions()

  rstart, rend = b.getOwnershipRange()
  b.setValues(range(rstart, rend), b_scipy[rstart:rend])
  b.assemble()

  return b


# -----------------------------
# Load matrices
# -----------------------------
data_folder = sys.argv[1]
print("Reading A")
A = read_mat_A(data_folder)
print("Reading b")
b = read_vec_b(data_folder)
print("Create x")
x = b.duplicate()
x.zeroEntries()

# -----------------------------
# Create linear solver KSP
# -----------------------------
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setTolerances(
  rtol=1e-8,
  atol=1e-12,
  divtol=1e5,
  max_it=10000
)

ksp.setType("gmres") # gmres, bcgs, qcg
pc = ksp.getPC()
pc.setType("gamg") # jacobi, ilu, gamg, hypre

ksp.setFromOptions()

# Sove the system
print("Start solving...")
ts = MPI.Wtime()
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


