"""
Poisson / Laplace 2D — high-level API example.

Steady linear solve with Dirichlet boundary conditions: inlet at 20, the three
other walls at 0.

Run:
    python poisson2d_api.py
    mpirun -n 4 python poisson2d_api.py
"""
from manapy.api import Mesh, PoissonModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128))

# Boundary conditions declared together with the field.
# Each patch: a type string, or (type, value) for value-carrying types.
P = mesh.field("P", bc={"in":     ("dirichlet", 20.0),
                        "out":    ("dirichlet", 0.0),
                        "upper":  ("dirichlet", 0.0),
                        "bottom": ("dirichlet", 0.0)})

model = PoissonModel(P, mesh, solver="mumps")   # or solver="petsc"
model.solve()
model.save("P")                                  # writes vtk_results/

print("P range: [%.6f, %.6f]" % (float(P.cell.min()), float(P.cell.max())))
