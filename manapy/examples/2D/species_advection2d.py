#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passive multispecies transport demo: a scalar blob advected by a uniform flow.

Exercises SpeciesTransport (Phase 3): N species mass fractions
Y_k carried as partial densities q_k = rho*Y_k, advected with a Rusanov flux
consistent with the bulk Euler density flux. Two species summing to 1 demonstrate
the sum-preservation and conservation properties; the blob translates at the flow
speed.

Run:
    MESH_DIR=../../../meshes/geo python3 species_advection2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.system import EulerSolver
from manapy.solvers.euler.species import SpeciesTransport
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  MESH_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'meshes', 'geo')

mesh_path = os.path.join(MESH_DIR, os.environ.get('MESH_FILE', 'carre.msh'))
domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells

gamma = 1.4
rho0, p0, U0 = 1.0, 10.0, 1.0

rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
rho.cell[:] = rho0
P.cell[:] = p0
rhou.cell[:] = rho0 * U0
rhov.cell[:] = 0.0
rhoE.cell[:] = p0 / (gamma - 1.0) + 0.5 * rho0 * U0 ** 2

solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4,
                     scheme="rusanov", bc="Neumann")

xc = cells.center[:, 0]
x0 = xc.min(); L = xc.max() - xc.min()
Y1 = 0.2 + 0.6 * np.exp(-((xc - (x0 + 0.3 * L)) / (0.08 * L)) ** 2)
Y2 = 1.0 - Y1
species = SpeciesTransport(solver, [Y1, Y2], names=["A", "B"], renormalize=False)

m0 = species.total_mass(0) + species.total_mass(1)
tfinal = float(os.environ.get('TFINAL', 0.3 * L / U0))

# uniform flow stays steady; advect species only (passive transport demo)
time = 0.0
niter = 0
while time < tfinal:
  d_t = solver.stepper()
  if time + d_t > tfinal:
    d_t = tfinal - time
  time += d_t
  species.advance(d_t)
  niter += 1

Y = species.mass_fractions()
summ = Y[0] + Y[1]
sum_err = COMM.allreduce(float(np.max(np.abs(summ - 1.0))), op=MPI.MAX)
m1 = species.total_mass(0) + species.total_mass(1)

if RANK == 0:
  print(f"cells={COMM.allreduce(domain.nbcells, op=MPI.SUM)} iters={niter} t={time:.4f}")
  print(f"  max|sum(Y)-1| = {sum_err:.2e}   species mass rel.change = {abs(m1-m0)/m0:.2e}")
  print(f"  blob advected ~ U0*t = {U0*time:.4f}")
