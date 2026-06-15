"""
Darcy 2D — high-level API example.

Solves the pressure field, derives the velocity from its gradient, and transports
a Gaussian tracer with that velocity (coupled loop hidden inside DarcyModel).

Run:
    python darcy2d_api.py
    mpirun -n 4 python darcy2d_api.py
"""
import numpy as np

from manapy.api import Mesh, DarcyModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128))

# Pressure: Dirichlet inlet/outlet, Neumann (no-flux) on the side walls.
P = mesh.field("P", bc={"in":     ("dirichlet", 2.0),
                        "out":    ("dirichlet", 0.0),
                        "upper":  "neumann",
                        "bottom": "neumann"})

# Tracer transported by the Darcy velocity.
ne = mesh.field("ne", init=lambda x, y, z: np.exp(-((x - 0.2) ** 2 + (y - 0.5) ** 2) / 0.01))

model = DarcyModel(P, mesh, tracer=ne, order=2, cfl=0.8)
model.run(T=0.25, output_every=10)

u, v = model.velocity
print("P range: [%.4f, %.4f]   max|u|=%.4f" % (
    float(P.cell.min()), float(P.cell.max()), float(np.abs(u.cell).max())))
