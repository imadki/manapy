"""
Advection 2D — high-level API example.

Transports a Gaussian blob with a constant velocity (2, 0).

Run:
    python advection2d_api.py
    mpirun -n 4 python advection2d_api.py
"""
import numpy as np

from manapy.api import Mesh, AdvectionModel

# Mesh generated on the fly (no .msh file needed); dim is auto-detected.
mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128), cell_type="quad")

# Field with an initial Gaussian profile.
phi = mesh.field("phi", 
                 init=lambda x, y, z: np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / 0.01),
                 limiter='vanalbada')

# Constant velocity (each component may be a number, a callable, or a Variable).
model = AdvectionModel(phi, mesh, velocity=(2.0, 0.0), cfl=0.8, order=2, scheme="upwind")
model.run(T=0.25, output_every=10)
