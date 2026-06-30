#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulent inflow demo (Phase 7): synthetic-turbulence shear-layer inlet.

Builds a shear-layer synthetic-turbulence inflow: a hyperbolic-tangent mean
velocity profile between two streams with superimposed white-noise fluctuations
at a prescribed turbulence intensity. Prints the recovered mean profile and the
fluctuation statistics over many realizations.

Run:
    MESH_DIR=../../../meshes/geo python3 turbulent_inflow2d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver
from manapy.solvers.euler.inflow import TurbulentInflow

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), 'carre.msh')
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)

gamma = 1.4
rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
rho.cell[:] = 1.0; P.cell[:] = 1e5; rhoE.cell[:] = P.cell / (gamma - 1.0)
solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme="rusanov", bc="Neumann")

u1, u2, I = 100.0, 40.0, 0.06
inflow = TurbulentInflow(solver, "in", u1=u1, u2=u2, p=1e5, T=300.0,
                         delta=0.1, y0=0.5, intensity=I, seed=1)

# sample statistics over many realizations
N = 3000
usum = np.zeros(inflow.nf); u2sum = np.zeros(inflow.nf)
for _ in range(N):
  _, ru, _, _, _ = inflow.state()
  u = ru / inflow.rho
  usum += u; u2sum += u * u
umean = usum / N
urms = np.sqrt(np.maximum(u2sum / N - umean ** 2, 0))

if RANK == 0:
  order = np.argsort(inflow.yf)
  print(f"inflow faces = {inflow.nf}   streams u1={u1} u2={u2} m/s   intensity I={I}")
  print(f"mean profile recovers tanh (max err {np.max(np.abs(umean - inflow.u_mean)):.3f} m/s)")
  print(f"fluctuation rms = {urms.mean():.3f} m/s   target I*dU = {I * abs(u1 - u2):.3f} m/s")
  print("y :  u_mean(tanh)  u_sampled")
  for j in order[::max(1, inflow.nf // 8)]:
    print(f"  {inflow.yf[j]:5.2f}    {inflow.u_mean[j]:7.2f}    {umean[j]:7.2f}")
