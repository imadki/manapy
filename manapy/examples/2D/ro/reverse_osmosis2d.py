#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transient reverse-osmosis feed channel with membrane fouling.

Run (serial)::

    python3 reverse_osmosis2d.py

Run (parallel)::

    mpirun -n 4 python3 reverse_osmosis2d.py

The channel mesh (``meshes/ro_channel.msh``) has the membrane on the ``bottom``
patch (y=0), the feed inlet on ``in`` (x=0) and the concentrate outlet on
``out`` (x=L).  The script time-marches the coupled salt-transport / membrane /
fouling system and prints the permeate flux, recovery and wall concentration as
the fouling layer builds up.
"""
import os
from mpi4py import MPI

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.ro import ReverseOsmosisSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..')
MESH_DIR = os.environ.get('MESH_DIR', os.path.join(BASE_DIR, 'meshes'))
mesh_path = os.path.join(MESH_DIR, 'ro_channel.msh')

domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)

FEED = 35.0   # seawater salinity [kg/m3]

# Salt concentration: feed at the inlet (dirichlet), zero-gradient elsewhere
# (the membrane removal is handled by the solver as a source term).
boundaries = {"in": "dirichlet", "out": "neumann",
              "upper": "neumann", "bottom": "neumann"}
values = {"in": FEED}

c = Variable(domain=domain, BC=boundaries, values_dict=values, name="salt")
u = Variable(domain=domain, name="u")
v = Variable(domain=domain, name="v")

# start the channel filled with feed water
c.cell[:] = FEED

# A low cross-flow / high-permeability regime so that concentration
# polarisation (salt build-up at the wall) is clearly visible alongside the
# fouling-driven flux decline.
solver = ReverseOsmosisSolver(
    c, vel=(u, v),
    feed_conc=FEED,
    U0=0.01, D=1.0e-8,
    A_w=8.0e-12, B_s=5.0e-8, dP=6.5e6,
    fouling=True, fouling_coeff=0.4,
)

if RANK == 0:
    print(f"Clean membrane resistance R_m = {solver.R_m:.3e} 1/m")
    print(f"Nominal permeation velocity   = {solver.Jw_nom*1e3*3600:.2f} LMH")
    print("Start RO time-marching ...")

hist = solver.run(nsteps=400, history_every=20, verbose=True)

if RANK == 0:
    print("\nSummary")
    print(f"  initial flux : {hist['flux_LMH'][0]:6.2f} LMH   "
          f"recovery {hist['recovery'][0]*100:5.2f}%   "
          f"wall conc {hist['cw_mean'][0]:.2f} kg/m3")
    print(f"  final   flux : {hist['flux_LMH'][-1]:6.2f} LMH   "
          f"recovery {hist['recovery'][-1]*100:5.2f}%   "
          f"wall conc {hist['cw_mean'][-1]:.2f} kg/m3")
    drop = (1 - hist['flux_LMH'][-1] / hist['flux_LMH'][0]) * 100
    print(f"  flux decline due to fouling : {drop:.1f}%   "
          f"(R_f/R_m = {hist['Rf_over_Rm'][-1]:.2f})")
