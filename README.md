# manapy

`manapy` is a Python framework for solving systems of conservation laws with the
**cell-centered finite-volume method** on unstructured, possibly hybrid, meshes
in 2D and 3D. The same solver code runs serially, across many **MPI** ranks, and
on **CUDA GPUs**, and a small high-level API lets you set up a simulation in a
few lines.

## Features

- Finite-volume operators on unstructured / hybrid / periodic 2D and 3D meshes.
- Unified **CPU (Numba) and GPU (CUDA)** backends — kernels are written once.
- Distributed **MPI** domain decomposition with halo exchange.
- Distributed linear solvers via **PETSc**, **MUMPS**, **Ginkgo**, or **SciPy**.
- Built-in solvers: advection, advection–diffusion, Burgers, diffusion / Poisson
  / Darcy, compressible Euler, shallow water, and shallow-water MHD.
- Standard mesh I/O through [`meshio`](https://github.com/nschloe/meshio) and VTK
  output for ParaView.

## Requirements

- Python 3.8+
- A working MPI runtime for `mpi4py`
- (optional) a CUDA toolchain for the GPU backend; PETSc / MUMPS for the
  corresponding linear solvers

## Install

```bash
python3 -m pip install .            # standard install
python3 -m pip install -e ".[dev]"  # editable install + test dependencies
```

Build a wheel locally:

```bash
python3 -m build
pip install dist/*.whl
```

See the [c_api readme](./manapy/c_api/README.md) (Section 7 — uploading to PyPI;
Section 4 — manylinux wheel build), and the backend setup guides:
[MUMPS](./tools/install_mumps4py.md), [PETSc](./tools/install_petsc4py.md).

## Quickstart

Advect a Gaussian blob on a 128×128 mesh and write VTK output:

```python
import numpy as np
from manapy.api import Mesh, AdvectionModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128), cell_type="quad")
phi = mesh.field("phi",
                 init=lambda x, y, z: np.exp(-((x - 0.25)**2 + (y - 0.5)**2) / 0.01),
                 limiter="vanalbada")
model = AdvectionModel(phi, mesh, velocity=(2.0, 0.0), cfl=0.8, order=2, scheme="upwind")
model.run(T=0.25, output_every=10)
```

Run it in parallel with `mpirun -n 4 python quickstart.py` — no code change
needed.

## High-level API

The `manapy.api` module exposes `Mesh` plus a set of physics models. A mesh can
be generated on the fly (`Mesh.rectangle`, see `manapy.api.meshgen`) or read from
a file, and `mesh.field(name, init=..., bc=...)` creates a solution field.

```python
from manapy.api import Mesh, DiffusionModel, PoissonModel, DarcyModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128))

# Advection–diffusion of a scalar
model = DiffusionModel(phi, mesh, velocity=(1.0, 0.0), Dxx=0.01, Dyy=0.01, cfl=0.8, order=2)

# Poisson problem with Dirichlet/Neumann boundary conditions, solved with MUMPS
P = mesh.field("P", bc={"in": ("dirichlet", 20.0), "out": ("dirichlet", 0.0)})
model = PoissonModel(P, mesh, solver="mumps")   # or solver="petsc"

# Darcy (potential) flow transporting a passive tracer
model = DarcyModel(P, mesh, tracer=ne, order=2, cfl=0.8)

model.run(T=0.25, output_every=10)
```

For problems beyond the shipped models, the lower-level solvers in
`manapy.solvers` expose the finite-volume operators directly; see
[`manapy/solvers/ADDING_A_SOLVER.md`](manapy/solvers/ADDING_A_SOLVER.md).

## Examples

Runnable 2D and 3D cases for every solver are in
[`manapy/examples`](manapy/examples). The `*_api.py` files use the high-level API;
the others drive the solvers directly. Mesh sources (gmsh `.geo`) are in
[`meshes/geo`](meshes/geo) — generate a mesh with, e.g.,
`gmsh -2 meshes/geo/uns_square.geo -o meshes/geo/uns_square.msh`.

## Tests

```bash
python3 -m pytest tests                 # serial
mpirun -n 4 python3 -m pytest tests     # parallel paths
```

## Documentation

The full documentation (installation, usage, API reference) is built with Sphinx
from the [`docs/`](docs) directory:

```bash
python3 -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to
report bugs, ask questions, and submit pull requests.

## License

`manapy` is released under the [MIT License](LICENCE).
