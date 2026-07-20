#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation harness for the FV / fv_corrected GPU assembly port.

Runs ONE backend per invocation (a GPU CUDA context and a CPU run should not
share a process) and dumps, for the Darcy/Laplacian test problem on carre.msh:

  <prefix>_row.npy / _col.npy / _data.npy   the assembled COO Laplacian
  <prefix>_rhs.npy                          the assembled RHS (rhs0)
  <prefix>_sol.npy                          P.cell after one linear solve

Usage:
  python3 validate_fv.py <cpu|gpu> <fv|fv_corrected> <out_prefix>

Then compare two prefixes with compare_fv.py (matrix + rhs + solution).
The matrix comparison is solver-independent and is the sharpest assembly check.
"""
import os
import sys

import numpy as np
from mpi4py import MPI

from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.solvers.ls import GinkgoDistributedSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# --------------------------------------------------------------------------- args
if len(sys.argv) != 4:
  if RANK == 0:
    print(__doc__)
  sys.exit(1)
backend_kind = sys.argv[1].lower()      # cpu | gpu
scheme = sys.argv[2].lower()            # fv | fv_corrected
prefix = sys.argv[3]
assert backend_kind in ("cpu", "gpu")
assert scheme in ("fv", "fv_corrected")

# --------------------------------------------------------------------------- mesh
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MESH_DIR = os.environ.get(
  "MESH_DIR", os.path.join(BASE_DIR, "..", "..", "..", "..", "meshes", "geo"))
mesh_path = os.path.join(MESH_DIR, "carre.msh")
dim = 2

# ------------------------------------------------------------------------ backend
if backend_kind == "gpu":
  from manapy.backends.gpu import GPUBackend
  be = GPUBackend(float_precision="float64", int_precision="int32", cache=True)
  be.init_stream()
  be.set_config(free=True)
  device = "cuda"
else:
  from manapy.backends.cpu import CPUBackend
  be = CPUBackend(float_precision="float64", int_precision="int32", cache=True)
  device = "cpu"   # Ginkgo reference/CPU executor

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal,
                              recreate=True, backend=be)
cells = domain.cells
nbcells = domain.nbcells

# --------------------------------------------------------------------------- BCs
Pinit = 2.0
boundaries = {"in": "dirichlet", "out": "dirichlet",
              "upper": "neumann", "bottom": "neumann"}
values = {"in": Pinit, "out": 0.0}
P = Variable(domain=domain, BC=boundaries, values_dict=values)

# Non-trivial initial P.cell so the fv_corrected deferred correction (which uses
# grad(P)) is actually exercised on the first solve. Identical on both backends.
_ne = np.zeros(nbcells); _u = np.zeros(nbcells)
_v = np.zeros(nbcells); _P = np.zeros(nbcells)
initialisation_gaussian_2d(_ne, _u, _v, _P, be.to_host(cells.center), Pinit)
be.copy(P.cell, _P)

# ------------------------------------------------------------------------ solver
# Same AMG config the user runs in production (BiCGSTAB + Multigrid/Pgm, Schwarz
# Jacobi smoother, Cg coarse). reuse_mtx=False -> a fresh solve with a logger.
amg_args = {
  "type": "solver::Bicgstab",
  "preconditioner": {
    "type": "solver::Multigrid", "max_levels": 10, "min_coarse_rows": 2,
    "mg_level": [{"type": "multigrid::Pgm", "deterministic": True}],
    "pre_smoother": [{"type": "solver::Ir", "relaxation_factor": 0.9,
      "solver": {"type": "preconditioner::Schwarz",
                 "local_solver": {"type": "preconditioner::Jacobi"}},
      "criteria": [{"type": "Iteration", "max_iters": 2}]}],
    "post_uses_pre": True,
    "coarsest_solver": {"type": "solver::Cg",
                        "criteria": [{"type": "Iteration", "max_iters": 4}]},
    "default_initial_guess": "zero",
    "criteria": [{"type": "Iteration", "max_iters": 1}]},
  "criteria": [{"type": "Iteration", "max_iters": 1000},
               {"type": "ResidualNorm", "reduction_factor": 1e-10}],
}
L = GinkgoDistributedSolver(domain=domain, var=P, device=device, scheme=scheme,
                            reuse_mtx=False, verbose=False, solver_args=amg_args)

# One linear solve (this triggers presolve -> assembly() internally).
L()

# ---------------------------------------------------------------------- dump
def h(a):
  """Bring any backend array (GPUArray/device or host ndarray) to host."""
  try:
    return np.asarray(be.to_host(a))
  except Exception:
    return np.asarray(a)

row = h(L._row).astype(np.int64)
col = h(L._col).astype(np.int64)
data = h(L._data).astype(np.float64)
rhs = h(L.rhs0).astype(np.float64)
sol = h(P.cell).astype(np.float64)

np.save(f"{prefix}_row.npy", row)
np.save(f"{prefix}_col.npy", col)
np.save(f"{prefix}_data.npy", data)
np.save(f"{prefix}_rhs.npy", rhs)
np.save(f"{prefix}_sol.npy", sol)

if RANK == 0:
  print(f"[validate_fv] backend={backend_kind} scheme={scheme} device={device}")
  print(f"  nnz(triplets)={row.size}  ncells={sol.size}")
  print(f"  |data|_1={np.abs(data).sum():.6e}  |rhs|_2={np.linalg.norm(rhs):.6e}"
        f"  |sol|_2={np.linalg.norm(sol):.6e}")
  print(f"  saved -> {prefix}_*.npy")
