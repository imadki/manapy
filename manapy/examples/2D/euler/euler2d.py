#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euler 2D benchmark: linear advection of a Gaussian density pulse (entropy wave).

With uniform pressure and velocity, the density perturbation is simply advected
at the flow speed without distortion, so the exact solution is a translated
Gaussian -- the same analytic field the Manapy `advec` solver and PyFR
(`euler` system, entropy-wave init) reproduce. This makes it a fair, mesh-shared
benchmark between the two codes.

Initial / exact field:
    rho(x,y,t) = 1 + 5*exp(-((x-0.2-a*t)^2 + (y-0.2)^2)/sigma^2)
    u = a = 2,  v = 0,  p = 1,  gamma = 1.4

Run:
    MESH_DIR=../../../../meshes/geo mpirun -n 1 python3 euler2d.py
"""
from mpi4py import MPI
import os
import timeit
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.system import EulerSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = os.environ.get('MESH_FILE', 'carre.msh')
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

start = timeit.default_timer()
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
end = timeit.default_timer()
tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

cells = domain.cells
nbcells = domain.nbcells

gamma = 1.4
a = 2.0          # advection speed (x)
sigma = 0.05
x0, y0 = 0.2, 0.2
p0 = 1.0
tfinal = 0.25
# Fixed time step (set DT=... to match PyFR's `scheme=euler, controller=none`);
# DT unset / 0 falls back to the CFL-limited adaptive step.
fixed_dt = float(os.environ.get('DT', '0') or 0.0)

# conservative state variables (BC handled by the solver's ghost kernel)
rho = Variable(domain=domain)
P = Variable(domain=domain)
rhou = Variable(domain=domain)
rhov = Variable(domain=domain)
rhoE = Variable(domain=domain)

order = int(os.environ.get('ORDER', '2'))
S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.5,
                order=order, scheme="rusanov", bc="Neumann")


def exact_rho(t):
  xc = cells.center[:, 0]
  yc = cells.center[:, 1]
  return 1.0 + 5.0 * np.exp(-((xc - x0 - a * t) ** 2 + (yc - y0) ** 2) / sigma ** 2)


# --- initialisation (entropy wave) ---
rho.cell[:] = exact_rho(0.0)
P.cell[:] = p0
rhou.cell[:] = rho.cell[:] * a
rhov.cell[:] = 0.0
rhoE.cell[:] = 0.5 * rho.cell[:] * a ** 2 + P.cell[:] / (gamma - 1.0)

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Computation ...")

time = 0.0
niter = 0
dt_nominal = fixed_dt
while time < tfinal:
  if fixed_dt > 0.0:
    d_t = fixed_dt
    S.dt = d_t
  else:
    d_t = S.stepper()
  if niter == 0 and fixed_dt <= 0.0:
    dt_nominal = d_t
  if time + d_t > tfinal:
    d_t = tfinal - time
    S.dt = d_t
  time += d_t

  S.compute_fluxes(t=time)
  S.compute_new_val()
  niter += 1

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# --- L2 error vs analytic (volume-weighted) ---
err = rho.cell[:] - exact_rho(time)
vol = cells.volume[:]
local_num = float(np.sum(vol * err * err))
local_den = float(np.sum(vol))
num = COMM.allreduce(local_num, op=MPI.SUM)
den = COMM.allreduce(local_den, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)
l2 = np.sqrt(num / den)

if RANK == 0:
  mode = "fixed" if fixed_dt > 0.0 else "cfl"
  print(f"mesh={filename} cells={glob_cells} iters={niter} dt={dt_nominal:.6e} ({mode}) "
        f"tfinal={time:.6f} walltime={walltime:.4f}s L2(rho)={l2:.6e}")
