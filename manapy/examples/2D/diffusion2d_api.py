"""
Diffusion 2D — API example sans fichier .msh
=============================================

Le maillage est généré à la volée par Mesh.rectangle().

Run:
    python diffusion2d_api.py
    mpirun -n 4 python diffusion2d_api.py
"""

import numpy as np
from manapy.api import Mesh, Field, DiffusionModel

# ---------------------------------------------------------------------------
# Maillage généré directement — aucun fichier .msh nécessaire
# ---------------------------------------------------------------------------
mesh = Mesh.rectangle(Lx=1.0, Ly=0.4, nx=40, ny=16)

# ---------------------------------------------------------------------------
# Champ avec conditions aux limites
# ---------------------------------------------------------------------------
phi = Field(mesh, name="phi",
            bc={"in":  ("dirichlet", 2.0),
                "out": ("dirichlet", 0.0)},
            init=lambda x, y, z: 2.0 * (1.0 - x))

u = Field(mesh, name="u", init=1.0)
v = Field(mesh, name="v", init=0.0)

# ---------------------------------------------------------------------------
# Modèle et simulation
# ---------------------------------------------------------------------------
model = DiffusionModel(phi, velocity=(u, v), Dxx=0.1, cfl=0.8)
model.run(T=0.25, output_every=50, output_dir="output_diffusion")
