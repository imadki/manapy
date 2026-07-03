"""
Advection-diffusion 2D — high-level API example.

Transports and diffuses a Gaussian blob.

Run:
    python diffusion2d_api.py
    mpirun -n 4 python diffusion2d_api.py
"""
import numpy as np

from manapy.api import Mesh, DiffusionModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128))

phi = mesh.field("phi", init=lambda x, y, z: np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / 0.01))

# Diffusion coefficients Dxx/Dyy in addition to the advective velocity.
model = DiffusionModel(phi, mesh, velocity=(1.0, 0.0), Dxx=0.01, Dyy=0.01, cfl=0.8, order=2)
model.run(T=0.25, output_every=10)
