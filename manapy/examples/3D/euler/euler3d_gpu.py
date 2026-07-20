#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euler 3D benchmark (GPU): linear advection of a Gaussian density pulse (entropy wave).

GPU twin of euler2d.py -- same entropy-wave setup extended to 3D, run on the CUDA
backend (GPUBackend + the kernels in manapy/solvers/euler/cuda_fvm_utils3d.py). With
uniform pressure and velocity the density perturbation is simply advected at the flow
speed without distortion, so the exact solution is a translated Gaussian. 1st-order
Rusanov (the 3D Euler kernels are order-1 only), Neumann ghost boundaries.

Initial / exact field:
    rho(x,y,z,t) = 1 + 5*exp(-((x-x0-a*t)^2 + (y-y0)^2 + (z-z0)^2)/sigma^2)
    u = a = 2,  v = w = 0,  p = 1,  gamma = 1.4

Run:
    MESH_DIR=../../../../meshes/geo MESH_FILE=cube.msh mpirun -n 1 python3 euler3d_gpu.py
"""
from mpi4py import MPI
import os
import timeit
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.system import EulerSolver
from manapy.core.Variable import Variable
from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = os.environ.get('MESH_FILE', 'cube.msh')
dim = 3
mesh_path = os.path.join(MESH_DIR, filename)

# --- GPU backend ---
gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=False)
gpu.init_stream()
gpu.set_config(free=True, nb_threads=64)   # 256 threads/bloc au lieu de 32

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
gamma = 1.4
a = 2.0          # advection speed (x)
sigma = 0.10
x0, y0, z0 = 0.2, 0.5, 0.5
p0 = 1.0
tfinal = float(os.environ.get('TFINAL', '0.20'))
# Fixed time step (set DT=... to match a controller=none run); DT unset / 0 falls back
# to the CFL-limited adaptive step.
fixed_dt = float(os.environ.get('DT', '0') or 0.0)

# conservative state variables (BC handled by the solver's ghost kernel)
rho = Variable(domain=domain)
P = Variable(domain=domain)
rhou = Variable(domain=domain)
rhov = Variable(domain=domain)
rhow = Variable(domain=domain)
rhoE = Variable(domain=domain)

# 3D Euler CUDA kernels are 1st-order only.
S = EulerSolver(rho, P, rhou, rhov, rhoE, rhow=rhow, gamma=gamma, cfl=0.5,
                order=1, scheme="rusanov", bc="Neumann")


def exact_rho(centers, t):
  xc = centers[:, 0]
  yc = centers[:, 1]
  zc = centers[:, 2]
  return 1.0 + 5.0 * np.exp(
      -((xc - x0 - a * t) ** 2 + (yc - y0) ** 2 + (zc - z0) ** 2) / sigma ** 2)


# --- initialisation (entropy wave) --- computed on host, pushed to the device Variables.
centers_h = np.asarray(gpu.to_host(cells.center))
rho_h = exact_rho(centers_h, 0.0)
P_h = np.full_like(rho_h, p0)
rhou_h = rho_h * a
rhov_h = np.zeros_like(rho_h)
rhow_h = np.zeros_like(rho_h)
rhoE_h = 0.5 * rho_h * a ** 2 + P_h / (gamma - 1.0)

for var, val in ((rho, rho_h), (P, P_h), (rhou, rhou_h), (rhov, rhov_h),
                 (rhow, rhow_h), (rhoE, rhoE_h)):
  gpu.copy(var.cell, gpu.to_device(np.ascontiguousarray(val, dtype=np.float64)))

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Computation ...")

time = 0.0
niter = 0

d_t = 6.954834e-05
while time < tfinal:
  S.dt = d_t
  time += d_t

  S.compute_fluxes(t=time)
  S.compute_new_val()
  niter += 1

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# --- L2 error vs analytic (volume-weighted, OWNED cells only) --- copied to host.
rho_hs = np.asarray(gpu.to_host(rho.cell))[:nbcells]
vol = np.asarray(gpu.to_host(cells.volume))[:nbcells]
cen = np.asarray(gpu.to_host(cells.center))[:nbcells]
err = rho_hs - exact_rho(cen, time)
local_num = float(np.sum(vol * err * err))
local_den = float(np.sum(vol))
num = COMM.allreduce(local_num, op=MPI.SUM)
den = COMM.allreduce(local_den, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)
l2 = np.sqrt(num / den)

if RANK == 0:
  mode = "fixed" if fixed_dt > 0.0 else "cfl"
  print(f"mesh={filename} cells={glob_cells} iters={niter} dt={d_t:.6e} ({mode}) "
        f"tfinal={time:.6f} walltime={walltime:.4f}s L2(rho)={l2:.6e}")
