#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reactive flow demo: constant-volume H2/air bomb with the coupled ReactiveSolver.

A closed box filled with a quiescent stoichiometric H2/air mixture is ignited by
its own kinetics. With no mean flow the hydrodynamics is trivial, so the coupled
solver must reproduce the 0-D constant-volume ignition: temperature and pressure
rise to the UV-equilibrium (adiabatic flame) values. This validates the Strang
operator-split coupling of hydro + species transport + real Cantera chemistry and
the real-EOS pressure feedback.

Needs `pip install cantera`. Uses a small mesh because the stiff source term is
integrated with a per-cell Cantera reactor.

Run:
    MESH_FILE=hybrid2d.msh python3 reactive_bomb2d.py
"""
from mpi4py import MPI
import os
import numpy as np
import cantera as ct

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver
from manapy.solvers.euler.cantera_backend import CanteraChemistry
from manapy.solvers.euler.reactive_solver import ReactiveSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', 'meshes')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), os.environ.get('MESH_FILE', 'hybrid2d.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)

# --- real H2/air chemistry (Cantera = open-source CHEMKIN) ---
chem = CanteraChemistry("h2o2.yaml")
Y = chem.mass_fractions_from(H2=2 * 2.016, O2=32.0, N2=3.76 * 28.0)
T0, p0 = 1100.0, ct.one_atm
gas = chem.gas
gas.TPY = T0, p0, Y
rho0, e0 = gas.density, gas.int_energy_mass
gamma_rep = gas.cp_mass / gas.cv_mass

# --- quiescent uniform initial state ---
rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
rho.cell[:] = rho0
P.cell[:] = p0
rhoE.cell[:] = rho0 * e0                       # zero kinetic energy

solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma_rep, cfl=0.4,
                     scheme="rusanov", bc="Neumann")
rs = ReactiveSolver(solver, chem, [Y[k] for k in range(chem.nspec)])

# 0-D reference (constant UV equilibrium)
Teq = chem.equilibrium_T(rho0, T0, Y)
gas.TDY = T0, rho0, Y; gas.equilibrate("UV"); Peq = gas.P


def mean_T():
  e = (rhoE.cell - 0.5 * (rhou.cell ** 2 + rhov.cell ** 2) / rho.cell) / rho.cell
  Yf = np.column_stack([q.cell / rho.cell for q in rs.species.q])
  return chem.eos_array(rho.cell, e, Yf)[0].mean()


if RANK == 0:
  print(f"cells={domain.nbcells}  init T={T0:.0f} P={p0:.0f}  -> 0-D reference Teq={Teq:.0f} Peq={Peq:.0f}")

t = 0.0
for it in range(80):
  dt = rs.step(t=t)
  t += dt
  if mean_T() > Teq - 50:
    break

if RANK == 0:
  YH2 = np.mean([q.cell / rho.cell for q in rs.species.q], axis=1)[chem.index("H2")]
  print(f"ignited at t={t:.3e} s")
  print(f"final  <T>={mean_T():.0f} K (eq {Teq:.0f})   <P>={P.cell.mean():.0f} Pa (eq {Peq:.0f})")
  print(f"       residual H2 mass fraction = {YH2:.4f}")
