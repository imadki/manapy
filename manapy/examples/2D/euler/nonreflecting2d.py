#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-reflecting (characteristic far-field) BC demo: radiating acoustic pulse.

A Gaussian pressure pulse in a quiescent gas expands outward and crosses the
domain boundaries. With the characteristic far-field BC ("NonReflecting") the
outgoing acoustic wave leaves with minimal spurious reflection -- the
unstructured-FV characteristic far-field (NSCBC) non-reflecting BC,
built on Riemann invariants in manapy's ghost-cell framework:
  - the outgoing acoustic invariant (un+c) is taken from the interior,
  - the incoming one (un-c) from the free-stream reference (rho_inf,u_inf,v_inf,p_inf),
  - entropy and the tangential velocity follow the local flow direction.

Set BC=Neumann (zero-gradient) to contrast the residual acoustic energy left in
the domain after the pulse has had time to exit.

Run:
    MESH_DIR=../../../../meshes/geo mpirun -n 1 python3 nonreflecting2d.py
    BC=Neumann python3 nonreflecting2d.py        # reflecting contrast
"""
from mpi4py import MPI
import os
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

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
nbcells = domain.nbcells

# --- quiescent free-stream reference (the far-field state) ---
gamma = 1.4
rho0 = 1.0
p0 = 1.0
c0 = np.sqrt(gamma * p0 / rho0)

bc = os.environ.get('BC', 'NonReflecting')

rho = Variable(domain=domain)
P = Variable(domain=domain)
rhou = Variable(domain=domain)
rhov = Variable(domain=domain)
rhoE = Variable(domain=domain)

# --- isentropic Gaussian pressure pulse at the domain centre ---
xc = cells.center[:, 0]
yc = cells.center[:, 1]
x0 = xc.mean()
y0 = yc.mean()
Lx = xc.max() - xc.min()
sigma = 0.05 * max(Lx, 1.0)
amp = 0.1 * p0

r2 = (xc - x0) ** 2 + (yc - y0) ** 2
p_init = p0 + amp * np.exp(-r2 / sigma ** 2)
rho.cell[:] = rho0 * (p_init / p0) ** (1.0 / gamma)
P.cell[:] = p_init
rhou.cell[:] = 0.0
rhov.cell[:] = 0.0
rhoE.cell[:] = P.cell[:] / (gamma - 1.0)

S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                order=1, scheme="rusanov", bc=bc,
                rho_inf=rho0, u_inf=0.0, v_inf=0.0, p_inf=p0)

vol = cells.volume[:]
e0_local = float(np.sum(vol * (P.cell[:] - p0) ** 2))
e0 = COMM.allreduce(e0_local, op=MPI.SUM)

# integrate ~2 acoustic crossing times -> a clean pulse should have left
tfinal = float(os.environ.get('TFINAL', 2.0 * Lx / c0))

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print(f"Start Computation (bc={bc}) ...")

time = 0.0
niter = 0
while time < tfinal:
  d_t = S.stepper()
  if time + d_t > tfinal:
    d_t = tfinal - time
    S.dt = d_t
  time += d_t
  S.compute_fluxes(t=time)
  S.compute_new_val()
  niter += 1

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

ef_local = float(np.sum(vol * (P.cell[:] - p0) ** 2))
ef = COMM.allreduce(ef_local, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)

if RANK == 0:
  print(f"bc={bc} cells={glob_cells} iters={niter} tfinal={time:.4f} "
        f"walltime={walltime:.4f}s  pulse_energy {e0:.4e} -> residual {ef:.4e} "
        f"(ratio {ef / e0:.5f})")
