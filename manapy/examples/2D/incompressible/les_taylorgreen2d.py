#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LES demo: decaying Taylor-Green vortex with a subgrid-scale model.

Exercises the LES extension of EulerSolver (Smagorinsky / WALE eddy viscosity,
in manapy's unstructured viscous path).
The resolved eddy viscosity mu_t adds to the laminar mu in the Newtonian stress;
the turbulent conductivity uses the turbulent Prandtl number Pr_t.

The Taylor-Green vortex is a standard LES/turbulence-decay benchmark. At low Mach
the kinetic energy decays monotonically; the SGS model supplies extra dissipation
where the resolved strain is large.

Run:
    MESH_DIR=../../../../meshes/geo python3 les_taylorgreen2d.py
    SGS=smagorinsky python3 les_taylorgreen2d.py
    SGS=wale        python3 les_taylorgreen2d.py
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
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
nbcells = domain.nbcells

# --- parameters ---
gamma = 1.4
R = 287.0
rho0 = 1.0
p0 = 100.0          # high p -> low Mach (nearly incompressible)
U0 = 1.0
mu = 1.0e-4         # small molecular viscosity -> SGS dominates the dissipation
Pr = 0.72
sgs_model = os.environ.get('SGS', 'wale')

rho = Variable(domain=domain)
P = Variable(domain=domain)
rhou = Variable(domain=domain)
rhov = Variable(domain=domain)
rhoE = Variable(domain=domain)

bc_vel = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                order=1, scheme="rusanov", bc="Neumann",
                viscous=True, mu=mu, Pr=Pr, R=R, viscosity_law="constant",
                cfl_visc=0.3, bc_vel=bc_vel, bc_temp=bc_vel,
                les=True, sgs_model=sgs_model, Cs=0.16, Cw=0.5, Prt=0.9)

# --- Taylor-Green initial field ---
xc = cells.center[:, 0]
yc = cells.center[:, 1]
k = 2.0 * np.pi
u = U0 * np.sin(k * xc) * np.cos(k * yc)
v = -U0 * np.cos(k * xc) * np.sin(k * yc)
rho.cell[:] = rho0
P.cell[:] = p0 + (rho0 * U0 ** 2 / 4.0) * (np.cos(2 * k * xc) + np.cos(2 * k * yc))
rhou.cell[:] = rho0 * u
rhov.cell[:] = rho0 * v
rhoE.cell[:] = P.cell[:] / (gamma - 1.0) + 0.5 * (rhou.cell[:] ** 2 + rhov.cell[:] ** 2) / rho.cell[:]

vol = cells.volume[:]


def kinetic_energy():
  ke = 0.5 * (rhou.cell[:] ** 2 + rhov.cell[:] ** 2) / rho.cell[:]
  loc = float(np.sum(vol * ke))
  return COMM.allreduce(loc, op=MPI.SUM)


tfinal = float(os.environ.get('TFINAL', 0.5))
ke0 = kinetic_energy()

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print(f"Start LES (sgs={sgs_model}) ...")

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

ke1 = kinetic_energy()
mut = S._mut.cell[:]
mut_max = COMM.allreduce(float(mut.max()), op=MPI.MAX)
mut_mean_loc = float(np.sum(vol * mut))
mut_mean = COMM.allreduce(mut_mean_loc, op=MPI.SUM) / COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)

if RANK == 0:
  print(f"sgs={sgs_model} cells={glob_cells} iters={niter} t={time:.4f} walltime={walltime:.3f}s")
  print(f"  kinetic energy {ke0:.6e} -> {ke1:.6e}  (decayed {100*(1-ke1/ke0):.2f}%)")
  print(f"  eddy viscosity mu_t: mean={mut_mean:.3e} max={mut_max:.3e}  (mu_lam={mu:.1e})")
