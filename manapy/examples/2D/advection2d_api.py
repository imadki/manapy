"""
Advection 2D — API example
==========================

Equivalent to advection2d.py but using the high-level manapy.api.

Run sequentially:
    python advection2d_api.py

Run in parallel:
    mpirun -n 4 python advection2d_api.py
"""

import os
import numpy as np

from manapy.api import Mesh, AdvectionModel
from manapy.ast import Variable
from manapy.solvers.advec.tools_utils import initialisation_gaussian_2d

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MESH_FILE = os.path.join(BASE_DIR, "mesh", "carre.msh")

mesh = Mesh(MESH_FILE, dim=2, backend="numba", cache=True)
domain = mesh.domain

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
Pinit = 2.0

ne = Variable(domain=domain, name="ne")
u  = Variable(domain=domain, name="u")
v  = Variable(domain=domain, name="v")
P  = Variable(domain=domain, name="P")

ne.cell[:] = 0.0
u.cell[:]  = 0.0
v.cell[:]  = 0.0
P.cell[:]  = 0.0

# Custom initialisation (Gaussian profile)
initialisation_gaussian_2d(
    ne.cell, u.cell, v.cell, P.cell,
    domain.cells.center, Pinit
)

# Constant velocity on faces (needed by the flux scheme)
u.face[:] = 2.0
v.face[:] = 0.0

# ---------------------------------------------------------------------------
# Model and run
# ---------------------------------------------------------------------------
model = AdvectionModel(
    ne, mesh,
    velocity=(u, v),
    cfl=0.8,
    order=1,
    output=[(ne, "ne"), (u, "u"), (v, "v"), (P, "P")],
)

model.run(T=0.25, output_every=50, output_dir="output_advection")
