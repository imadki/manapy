# High-level models

The `manapy.api` module exposes a `Mesh` class and a set of physics models. A
mesh is generated on the fly (`Mesh.rectangle`, see `manapy.api.meshgen`) or read
from a file, and `mesh.field(name, init=..., bc=...)` creates a solution field.
Every model exposes a `.run(T=..., output_every=...)` method that advances the
solution and writes VTK output.

## Advection

```python
from manapy.api import Mesh, AdvectionModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128), cell_type="quad")
phi = mesh.field("phi", init=lambda x, y, z: ..., limiter="vanalbada")
model = AdvectionModel(phi, mesh, velocity=(2.0, 0.0), cfl=0.8, order=2, scheme="upwind")
model.run(T=0.25, output_every=10)
```

## Advection–diffusion

```python
from manapy.api import Mesh, DiffusionModel

model = DiffusionModel(phi, mesh, velocity=(1.0, 0.0),
                       Dxx=0.01, Dyy=0.01, cfl=0.8, order=2)
model.run(T=0.25, output_every=10)
```

## Poisson

```python
from manapy.api import Mesh, PoissonModel

P = mesh.field("P", bc={"in": ("dirichlet", 20.0), "out": ("dirichlet", 0.0)})
model = PoissonModel(P, mesh, solver="mumps")   # or solver="petsc"
model.run()
```

## Darcy flow with a passive tracer

```python
import numpy as np
from manapy.api import Mesh, DarcyModel

P = mesh.field("P", bc={"in": ("dirichlet", 2.0), "out": ("dirichlet", 0.0)})
ne = mesh.field("ne", init=lambda x, y, z: np.exp(-((x - 0.2)**2 + (y - 0.5)**2) / 0.01))
model = DarcyModel(P, mesh, tracer=ne, order=2, cfl=0.8)
model.run(T=0.25, output_every=10)
```

## Beyond the built-in models

For conservation laws not covered by the shipped models, the lower-level solvers
in `manapy.solvers` expose the finite-volume operators (fluxes, gradients,
boundary conditions, linear solves) directly. See the guide
[`manapy/solvers/ADDING_A_SOLVER.md`](https://github.com/imadki/manapy/blob/main/manapy/solvers/ADDING_A_SOLVER.md)
and use the existing solvers (`advec`, `diffusion`, `euler`, `shallowater`, …) as
templates.
