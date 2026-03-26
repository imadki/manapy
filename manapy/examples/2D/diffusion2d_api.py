"""
Diffusion 2D — API example sans fichier .msh
=============================================

Le maillage est généré à la volée par Mesh.generate().

Run:
    python diffusion2d_api.py
    mpirun -n 4 python diffusion2d_api.py
"""

import numpy as np
from manapy.api import Mesh, DiffusionModel
from manapy.ast import Variable

# ---------------------------------------------------------------------------
# Maillage généré directement — aucun fichier .msh nécessaire
# ---------------------------------------------------------------------------
mesh = Mesh.generate(dim=2, bounds=((0, 1), (0, 0.4)), n=(40, 16))
domain = mesh.domain

# ---------------------------------------------------------------------------
# Variables avec conditions aux limites
# ---------------------------------------------------------------------------
c = domain.cells.center
phi = Variable(domain=domain, name="phi")
phi.cell[:] = 2.0 * (1.0 - c[:, 0])

u = Variable(domain=domain, name="u")
v = Variable(domain=domain, name="v")
u.cell[:] = 1.0
v.cell[:] = 0.0
u.face[:] = 1.0
v.face[:] = 0.0

# ---------------------------------------------------------------------------
# Modèle et simulation
# ---------------------------------------------------------------------------
model = DiffusionModel(phi, mesh, velocity=(u, v), Dxx=0.1, cfl=0.8)
model.run(T=0.25, output_every=50, output_dir="output_diffusion")
