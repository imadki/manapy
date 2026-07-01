#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D multi-gamma contact with the double-flux method (pressure-equilibrium preserving).

Two gases with different ratios of specific heats (gamma_L=1.6, gamma_R=1.2) share a
smooth contact that is advected at a uniform velocity. Physically the pressure must
stay uniform across a contact discontinuity; a naive variable-gamma scheme generates
a spurious pressure oscillation there, while the double-flux update (Abgrall-Billet),
which freezes each cell's gamma and re-syncs rhoE = P/(gamma-1)+KE, keeps it in exact
pressure equilibrium. This is the 3D analogue of examples/2D and the building block
for 3D variable-gamma combustion (the ReactiveSolver works in 3D unchanged).

The example runs both schemes and reports the L2 pressure error vs the uniform P0
(double-flux -> machine precision, variable-gamma alone -> O(1e-2)), and writes VTK
of the density, gamma and the pressure error for both.

Run:
    MESH_DIR=../../../meshes mpirun -n 1 python3 multigamma_contact3d.py
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver
from manapy.solvers.euler.species import SpeciesTransport

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), os.environ.get('MESH_FILE', 'hybrid3d.msh'))
domain = Domain.create_domain(mesh, 3, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc, vol = cells.center[:, 0], np.asarray(cells.volume)
L = xc.max() - xc.min()

gL, gR, P0, u0 = 1.6, 1.2, 1.0, 0.5
Y0 = 0.5 * (1 + np.tanh((xc - (xc.min() + 0.4 * L)) / (0.05 * L)))     # smooth contact in x


def gamma_of(Y):
  return 1.0 + 1.0 / ((1 - Y) / (gL - 1) + Y / (gR - 1))


def run(doubleflux, tag):
  g_init = gamma_of(Y0)
  rhoc = 1.0 + Y0
  rho, P, rhou, rhov, rhoE = (Variable(domain=domain) for _ in range(5))
  rhow = Variable(domain=domain)
  rho.cell[:] = rhoc; P.cell[:] = P0
  rhou.cell[:] = rhoc * u0
  rhoE.cell[:] = P0 / (g_init - 1) + 0.5 * rhoc * u0 ** 2
  S = EulerSolver(rho, P, rhou, rhov, rhoE, rhow=rhow, gamma=1.4, cfl=0.3,
                  scheme="rusanov", bc="Neumann", variable_gamma=True, doubleflux=doubleflux)
  sp = SpeciesTransport(S, [1 - Y0, Y0], renormalize=True)
  S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
  t = 0.0
  for _ in range(int(os.environ.get('NSTEP', 60))):
    dt = S.stepper()
    t += dt
    S.compute_fluxes(t); S.compute_new_val(); sp.advance(dt)
    S.set_gamma(gamma_of(sp.q[1].cell / rho.cell))
  perr = np.abs(P.cell - P0)
  l2 = float(np.sqrt(np.sum(vol * perr ** 2) / np.sum(vol)))
  if int(os.environ.get('OUTPUT', 1)):
    domain.save_on_cell_multi([f"rho_{tag}", f"gamma_{tag}", f"perr_{tag}"],
                              [rho.cell.copy(), S.gamma_cell.copy(), perr], dt, t, 0, 0)
  return l2, t


l2_df, t = run(True, "df")
l2_vg, _ = run(False, "vg")
if RANK == 0:
  print(f"3D multi-gamma contact (double-flux), t={t:.3f}")
  print(f"  L2(P - P0):  double-flux = {l2_df:.3e}   variable-gamma = {l2_vg:.3e}")
  print(f"  -> double-flux keeps pressure equilibrium {l2_vg / max(l2_df, 1e-30):.1e}x tighter")
  if int(os.environ.get('OUTPUT', 1)):
    print("  wrote VTK (rho, gamma, perr for df & vg) to ./vtk_results/")
