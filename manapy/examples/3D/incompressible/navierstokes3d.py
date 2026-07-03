#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navier-Stokes 3D benchmark: viscous decay of a sinusoidal shear layer.

This exercises the compressible viscous (Newtonian stress + Fourier conduction)
compressible viscous fluxes in manapy's EulerSolver -- the
3D extension of examples/2D/navierstokes2d.py.

Exact unsteady solution (constant mu, uniform rho and p):
    u(z,t) = U0 * sin(pi*z) * exp(-nu * pi^2 * t),   v = w = 0,   nu = mu/rho0
    rho = rho0,   p = p0  (uniform)

It is an exact Navier-Stokes solution because the convective term vanishes
(u depends on z only and v=w=0) and the pressure is uniform; only the viscous
diffusion of x-momentum remains, giving the exponential decay above. The walls
z=0 (back) and z=1 (front) are no-slip (u=0, since sin(pi*0)=sin(pi*1)=0); the
remaining sides are zero-gradient (du/dn=0). A low Mach number (c >> U0) keeps
it nearly incompressible.

The bundled mesh meshes/hybrid3d.msh is the box x[0,2] y[0,1] z[0,1] with named
boundaries in/out (x), upper/bottom (y), front/back (z).

Run:
    mpirun -n 1 python3 navierstokes3d.py
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
  MESH_DIR = os.path.join(BASE_DIR, 'meshes')

filename = os.environ.get('MESH_FILE', 'hybrid3d.msh')
dim = 3
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
nbcells = domain.nbcells

# --- physical parameters ---
gamma = 1.4
R = 287.0
rho0 = 1.0
p0 = 10.0            # large -> c = sqrt(gamma p/rho) >> U0  (low Mach)
U0 = 0.05
mu = 0.1
Pr = 0.72
nu = mu / rho0
# Viscous decay time of the sin(pi*z) mode. The explicit viscous stability limit
# forces a small dt on fine meshes, so by default we integrate a fraction of tau
# (partial decay) -- still an exact-solution check, but fast. Set TFINAL=... (e.g.
# the full tau) to run longer.
tau = 1.0 / (nu * np.pi ** 2)
tfinal = float(os.environ.get('TFINAL', 0.1 * tau))

# conservative state (inviscid BC handled by the solver's ghost kernel)
rho = Variable(domain=domain)
P = Variable(domain=domain)
rhou = Variable(domain=domain)
rhov = Variable(domain=domain)
rhow = Variable(domain=domain)
rhoE = Variable(domain=domain)

# Primitive boundary treatment for the viscous (diamond) gradients:
#   z-walls (front/back) -> no-slip: u = v = w = 0 (dirichlet)
#   x/y sides            -> fully developed: zero-gradient (neumann)
#   temperature          -> adiabatic everywhere (neumann)
bc_vel = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann",
          "front": "dirichlet", "back": "dirichlet"}
vel_values = {"front": 0.0, "back": 0.0}
bc_temp = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann",
           "front": "neumann", "back": "neumann"}

S = EulerSolver(rho, P, rhou, rhov, rhoE, rhow=rhow, gamma=gamma, cfl=0.5,
                order=1, scheme="rusanov", bc="Neumann",
                viscous=True, mu=mu, Pr=Pr, R=R, viscosity_law="constant",
                cfl_visc=0.3,
                bc_vel=bc_vel, bc_temp=bc_temp, vel_values=vel_values)


def exact_u(t):
  zc = cells.center[:, 2]
  return U0 * np.sin(np.pi * zc) * np.exp(-nu * np.pi ** 2 * t)


# --- initialisation ---
rho.cell[:] = rho0
P.cell[:] = p0
rhou.cell[:] = rho0 * exact_u(0.0)
rhov.cell[:] = 0.0
rhow.cell[:] = 0.0
rhoE.cell[:] = P.cell[:] / (gamma - 1.0) \
    + 0.5 * (rhou.cell[:] ** 2 + rhov.cell[:] ** 2 + rhow.cell[:] ** 2) / rho.cell[:]

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Computation ...")

time = 0.0
niter = 0
dt_nominal = 0.0
while time < tfinal:
  d_t = S.stepper()
  if niter == 0:
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

# --- L2 error of u vs analytic (volume-weighted) ---
u_num = rhou.cell[:] / rho.cell[:]
err = u_num - exact_u(time)
vol = cells.volume[:]
local_num = float(np.sum(vol * err * err))
local_den = float(np.sum(vol))
num = COMM.allreduce(local_num, op=MPI.SUM)
den = COMM.allreduce(local_den, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)
l2 = np.sqrt(num / den)

if RANK == 0:
  print(f"mesh={filename} cells={glob_cells} iters={niter} dt={dt_nominal:.6e} "
        f"tfinal={time:.6f} (tau={tau:.4f}) walltime={walltime:.4f}s L2(u)={l2:.6e}")
