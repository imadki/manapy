#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reactive-transport demo (Phase 6): differential species diffusion.

A localized H2 perturbation in air diffuses with the *mixture-averaged* diffusion
coefficients from Cantera (the open-source EGlib/CHEMKIN transport). Because H2 is
light it spreads markedly faster than the heavier species -- the differential
diffusion that matters for flame structure. Uses the Fickian diffusion added to
SpeciesTransport; validated separately against the analytic Gaussian-spreading
rate (variance grows as 2 D t).

Run (needs cantera):
    MESH_DIR=../../../meshes/geo python3 species_diffusion2d.py
"""
from mpi4py import MPI
import os
import numpy as np
import cantera as ct

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver
from manapy.solvers.euler.species import SpeciesTransport
from manapy.solvers.euler.cantera_backend import CanteraChemistry

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.path.join(os.environ.get('MESH_DIR', BASE), os.environ.get('MESH_FILE', 'carre.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
xc = cells.center[:, 0]
vol = cells.volume

chem = CanteraChemistry("h2o2.yaml")
gamma = 1.4
rho = Variable(domain=domain); P = Variable(domain=domain)
rhou = Variable(domain=domain); rhov = Variable(domain=domain); rhoE = Variable(domain=domain)
rho.cell[:] = 1.0; P.cell[:] = ct.one_atm; rhoE.cell[:] = P.cell / (gamma - 1.0)
solver = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme="rusanov", bc="Neumann")

# air background with a localized H2 bump (H2 displaces N2 locally)
iH2, iO2, iN2 = chem.index("H2"), chem.index("O2"), chem.index("N2")
x0 = xc.mean(); L = xc.max() - xc.min()
bump = 0.2 * np.exp(-((xc - x0) ** 2) / (2 * (0.06 * L) ** 2))
Y0 = [np.zeros(domain.nbcells) for _ in range(chem.nspec)]
Y0[iH2] = bump
Y0[iO2] = 0.233 * (1 - bump)
Y0[iN2] = 0.767 * (1 - bump)
sp = SpeciesTransport(solver, Y0, names=chem.names, renormalize=False)


def width(k):
  Y = np.clip(sp.q[k].cell / rho.cell, 0, None)
  if Y.sum() == 0:
    return 0.0
  xb = np.sum(vol * Y * xc) / np.sum(vol * Y)
  return np.sqrt(np.sum(vol * Y * (xc - xb) ** 2) / np.sum(vol * Y))


# mixture-averaged D_k from Cantera (constant background state here)
T = P.cell / (rho.cell * (ct.gas_constant / chem.gas.mean_molecular_weight))
_, _, Dk = chem.transport_array(np.full(domain.nbcells, 1000.0), P.cell,
                                np.column_stack([sp.q[k].cell / rho.cell for k in range(chem.nspec)]))
Dmax = Dk.max()
h2 = vol.mean()
dt = 0.05 * h2 / Dmax          # explicit diffusion stability limit
nt = int(os.environ.get('NT', 400))

w_h2_0, w_n2_0 = width(iH2), width(iN2)
for it in range(nt):
  sp.diffuse(dt, [Dk[:, k] for k in range(chem.nspec)])

if RANK == 0:
  print(f"cells={domain.nbcells}  steps={nt}  dt={dt:.2e}")
  print(f"  D(H2)={Dk[0, iH2]:.3e}  D(N2)={Dk[0, iN2]:.3e}  m^2/s  (ratio {Dk[0, iH2] / Dk[0, iN2]:.1f})")
  print(f"  H2 cloud width {w_h2_0:.4f} -> {width(iH2):.4f}   (spread {width(iH2) - w_h2_0:.4f})")
  print(f"  N2 cloud width {w_n2_0:.4f} -> {width(iN2):.4f}   (spread {width(iN2) - w_n2_0:.4f})")
  print("  -> H2 diffuses faster than N2 (differential diffusion)")
