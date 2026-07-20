#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advection-diffusion 2D benchmark (GPU): transported + diffusing Gaussian pulse.

GPU twin of the flat euler3d_gpu.py, for the scalar advection-diffusion case, run on
the CUDA backend (GPUBackend + the kernels in manapy/solvers/advecdiff/cuda_fvm_utils.py).
Constant velocity a=(VX,VY) advects the pulse while diffusion D spreads it, so the exact
solution is a translated + widened Gaussian (heat kernel), amplitude decaying as S0/var.

Initial / exact field  (var = S0 + 2*D*t,  S0 = SIGMA0^2):
    u(x,y,t) = (S0/var) * exp(-((x-(X0+VX t))^2 + (y-(Y0+VY t))^2) / (2 var))
    a = (VX,VY) = (1,0),  D = 0.002,  order 1 (upwind convection + explicit diffusion)

Run:
    MESH_DIR=../../../../meshes/geo MESH_FILE=mesh_707.msh mpirun -n 1 python3 advdiff2d_gpu.py
"""
from mpi4py import MPI
import os
import timeit
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = os.environ.get('MESH_FILE', 'mesh_707.msh')
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

# --- GPU backend ---
gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=False)
gpu.init_stream()
gpu.set_config(free=True, nb_threads=128)   # 64 threads/bloc au lieu de 32

backend = gpu

start = timeit.default_timer()
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True,
                              backend=backend)
end = timeit.default_timer()
tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

cells = domain.cells
nbcells = domain.nbcells

# physical parameters
vx = 1.0
vy = 0.0
D = 0.002        # diffusion coefficient
sigma0 = 0.10
x0, y0 = 0.25, 0.5
cfl = 0.8
order = 1
tfinal = float(os.environ.get('TFINAL', '0.30'))
# Fixed time step (set DT=... to match a controller=none run); DT unset / 0 falls back
# to the CFL-limited adaptive step.
fixed_dt = float(os.environ.get('DT', '0') or 0.0)

s0 = sigma0 ** 2

# scalar field (BC handled by the solver's ghost kernel)
u = Variable(domain=domain, BC={"in": "neumann", "out": "neumann",
                                "upper": "neumann", "bottom": "neumann"}, name="u")

# constant velocity field: set on faces (device), interpolate to cells
u_vel = Variable(domain=domain, name="u_vel")
v_vel = Variable(domain=domain, name="v_vel")
gpu.assign(u_vel.face, vx)
gpu.assign(v_vel.face, vy)
u_vel.interpolate_facetocell()
v_vel.interpolate_facetocell()

# order-1 upwind convection + explicit diffusion (2D advec-diff CUDA kernels).
S = AdvectionDiffusionSolver(u, vel=(u_vel, v_vel), Dxx=D, Dyy=D,
                             order=order, cfl=cfl)


def exact(centers, t):
  xc = centers[:, 0]
  yc = centers[:, 1]
  var = s0 + 2.0 * D * t
  return (s0 / var) * np.exp(
      -(((xc - (x0 + vx * t)) ** 2 + (yc - (y0 + vy * t)) ** 2)) / (2.0 * var))


# --- initialisation (Gaussian pulse) --- computed on host, pushed to the device Variable.
centers_h = np.asarray(gpu.to_host(cells.center))
ic_h = exact(centers_h, 0.0)
gpu.copy(u.cell, gpu.to_device(np.ascontiguousarray(ic_h, dtype=np.float64)))

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Computation ...")

time = 0.0
niter = 0

d_t = 2.297640e-05
while time < tfinal:
  S.dt = d_t
  time += d_t

  S.compute_fluxes()
  S.compute_new_val()
  niter += 1

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# --- L2 error vs analytic (area-weighted, OWNED cells only) --- copied to host.
u_hs = np.asarray(gpu.to_host(u.cell))[:nbcells]
vol = np.asarray(gpu.to_host(cells.volume))[:nbcells]
cen = np.asarray(gpu.to_host(cells.center))[:nbcells]
err = u_hs - exact(cen, time)
local_num = float(np.sum(vol * err * err))
local_den = float(np.sum(vol))
num = COMM.allreduce(local_num, op=MPI.SUM)
den = COMM.allreduce(local_den, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)
l2 = np.sqrt(num / den)

if RANK == 0:
  mode = "fixed" if fixed_dt > 0.0 else "cfl"
  print(f"mesh={filename} cells={glob_cells} iters={niter} dt={d_t:.6e} ({mode}) "
        f"tfinal={time:.6f} walltime={walltime:.4f}s L2={l2:.6e}")