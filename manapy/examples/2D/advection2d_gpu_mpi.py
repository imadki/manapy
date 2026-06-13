import os
os.environ.setdefault("MANAPY_COMPILE_SYNC", "current")

import numpy as np
from mpi4py import MPI

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advec.system import AdvectionSolver
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "..", "..", "..")
mesh_path = os.path.join(BASE_DIR, "meshes", "geo", "carre.msh")

dim = 2
NITER = 5
saving_at_node = 0

if RANK == 0:
  print("SIZE =", SIZE)


def make(domain):
  cells = domain.cells

  ne = Variable(domain=domain)
  u = Variable(domain=domain)
  v = Variable(domain=domain)
  P = Variable(domain=domain)

  S = AdvectionSolver(ne, vel=(u, v), order=2, cfl=0.8)

  initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, 2.0)

  return ne, u, v, P, S


# ---------------- CPU ----------------
domain_cpu = Domain.create_domain(
  mesh_path,
  dim,
  Partitioning.Par_Nodal,
  recreate=True,
)

ne, u, v, P, S = make(domain_cpu)

for _ in range(NITER):
  u.face[:] = 2.0
  v.face[:] = 0.0

  u.interpolate_facetocell()
  v.interpolate_facetocell()

  S.stepper()
  S.compute_fluxes()
  S.compute_new_val()

ne_cpu = np.array(ne.cell, dtype=np.float64).copy()

print(
  f"[rank {RANK}] CPU done. "
  f"nbcells={domain_cpu.nbcells} halofaces={len(domain_cpu.halofaces)}"
)


# ---------------- GPU ----------------
gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=False)
gpu.init_stream()
gpu.set_config(free=True)

domain_gpu = Domain.create_domain(
  mesh_path,
  dim,
  Partitioning.Par_Nodal,
  recreate=True,
  backend=gpu,
)

ne, u, v, P, S = make(domain_gpu)

time = 0.0
d_t = 0.0
miter = 0

for niter in range(1, NITER + 1):
  gpu.assign(u.face, 2.0)
  gpu.assign(v.face, 0.0)

  u.interpolate_facetocell()
  v.interpolate_facetocell()

  d_t = S.stepper()
  time += d_t
  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter == NITER:
    ne_cell = np.asarray(ne.cell.to_host())
    u_cell = np.asarray(u.cell.to_host())
    v_cell = np.asarray(v.cell.to_host())
    p_cell = np.asarray(P.cell.to_host())

    if saving_at_node:
      raise NotImplementedError("GPU node saving is not implemented in this short MPI check")
    domain_gpu.save_on_cell_multi(["ne", "u", "v", "P"], [ne_cell, u_cell, v_cell, p_cell], d_t, time, niter, miter)
    miter += 1

ne_gpu = np.asarray(ne.cell.to_host(), dtype=np.float64)

local_max = float(np.abs(ne_cpu - ne_gpu).max())
global_max = COMM.reduce(local_max, op=MPI.MAX, root=0)

print(f"[rank {RANK}] local max|CPU-GPU| = {local_max:.3e}")

if RANK == 0:
  print(f"==> GLOBAL max|CPU-GPU| over {SIZE} ranks = {global_max:.3e}")
  assert global_max < 1e-6, "GPU != CPU en multi-rang"
  print(f"==> MULTI-RANK ({SIZE}) ADVECTION2D GPU == CPU : PASSED")
