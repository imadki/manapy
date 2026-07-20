---
title: 'manapy: a Python framework for finite-volume methods on unstructured meshes with unified CPU/GPU execution'
tags:
  - Python
  - finite volume method
  - computational fluid dynamics
  - unstructured meshes
  - high-performance computing
  - MPI
  - GPU
authors:
  # TODO: confirm author list, order, ORCIDs and affiliations before submission
  - name: Imad Kissami
    orcid: 0000-0000-0000-0000   # TODO
    corresponding: true
    affiliation: 1
  - name: Ayoub Ben Hamou
    orcid: 0000-0000-0000-0000   # TODO
    affiliation: 1
  - name: Mouad Haikal
    orcid: 0000-0000-0000-0000   # TODO
    affiliation: 1
affiliations:
  - name: "Mohammed VI Polytechnic University (UM6P), Morocco"   # TODO: confirm
    index: 1
date: 20 July 2026   # TODO: update at submission
bibliography: paper.bib
---

# Summary

`manapy` is a Python framework for solving systems of conservation laws with the
cell-centered finite-volume method on unstructured, possibly hybrid, meshes in
two and three dimensions. It provides the building blocks shared by most
finite-volume solvers — mesh partitioning, halo exchange, gradient and Laplacian
reconstruction, numerical fluxes, boundary conditions, and distributed linear
solvers — together with a high-level API through which a new physical model is
assembled by reusing these operators rather than re-implementing them. The same
solver code runs on a single core, across many MPI ranks, and on CUDA GPUs,
because the numerical kernels are written against a backend abstraction that is
compiled just-in-time for the target device [@lam2015numba]. Out of the box
`manapy` ships solvers for linear advection, advection–diffusion, the inviscid
Burgers equation, diffusion/Poisson/Darcy problems, the compressible Euler
equations, the shallow-water equations, and shallow-water magnetohydrodynamics,
and reads standard mesh formats through `meshio` [@meshio].

A complete 2D advection simulation, from mesh generation to time integration and
VTK output, is set up in a few lines through the high-level API:

```python
import numpy as np
from manapy.api import Mesh, AdvectionModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128), cell_type="quad")
phi = mesh.field("phi",
                 init=lambda x, y, z: np.exp(-((x - 0.25)**2 + (y - 0.5)**2) / 0.01),
                 limiter="vanalbada")
model = AdvectionModel(phi, mesh, velocity=(2.0, 0.0), cfl=0.8, order=2, scheme="upwind")
model.run(T=0.25, output_every=10)      # runs serially or under mpirun -n N
```

# Statement of need

Research in computational fluid dynamics and related fields routinely requires
prototyping new conservation laws, numerical fluxes, or boundary treatments and
then running them at scale. Established finite-volume codes such as OpenFOAM
[@weller1998] deliver production performance but are large C++ frameworks with a
steep learning curve for method development, while Python packages such as FiPy
[@guyer2009fipy] and PyClaw [@ketcheson2012pyclaw] make experimentation easy but
run on logically-structured grids or on the CPU only and do not move a single
implementation transparently between distributed CPU and GPU execution on general
unstructured meshes. GPU-resident frameworks such as PyFR [@witherden2014pyfr]
achieve excellent performance but target high-order flux-reconstruction schemes
on specific element types rather than a general finite-volume programming model.

`manapy` targets the gap between these: a Python-level programming model in which
a solver is expressed once, in terms of reusable finite-volume operators, and
then executed either through MPI domain decomposition on CPUs or on a GPU,
without rewriting the kernels. This is enabled by three design choices:

- **Unified CPU/GPU backends.** Field data and kernels are written against a
  backend interface; the CPU backend compiles kernels with Numba
  [@lam2015numba] and the GPU backend maps them to CUDA, so the numerics are
  authored once and dispatched to the active device.
- **A high-level, extensible, multi-physics API.** `Mesh`, `AdvectionModel`,
  `DiffusionModel`, `PoissonModel` and `DarcyModel` let users set up a
  simulation in a few lines, while the underlying solver machinery is shared —
  adding a new equation reuses the existing flux, gradient and linear-solve
  operators (documented in `manapy/solvers/ADDING_A_SOLVER.md`).
- **Scalable parallelism.** Meshes are partitioned and coupled through halo
  exchange over `mpi4py` [@dalcin2011mpi4py]; implicit problems are assembled
  and solved with distributed linear solvers via PETSc [@petsc], MUMPS [@mumps],
  Ginkgo [@ginkgo], or SciPy [@virtanen2020scipy].

The framework also implements advanced finite-volume ingredients — higher-order
(WENO) reconstruction, well-balanced treatments of source terms, and support for
hybrid unstructured 2D/3D grids — which makes it suitable both as a teaching and
prototyping tool and as a basis for research on new numerical methods. `manapy`
has been used to develop solvers ranging from scalar transport to shallow-water
magnetohydrodynamics, showing that the same infrastructure spans a broad range of
hyperbolic and elliptic problems.

# State of the field

|                              | manapy | OpenFOAM | FiPy | PyClaw | PyFR |
|------------------------------|:------:|:--------:|:----:|:------:|:----:|
| Language                     | Python |   C++    | Python | Python | Python |
| Numerical method             | FV     | FV       | FV   | FV     | FR   |
| General unstructured meshes  | ✓      | ✓        | ✓    | ✗      | ✓    |
| Hybrid 2D/3D elements        | ✓      | ✓        | ~    | ✗      | ✓    |
| Distributed (MPI)            | ✓      | ✓        | ✓    | ✓      | ✓    |
| GPU execution                | ✓      | ✗        | ✗    | ✗      | ✓    |
| Same solver code CPU + GPU   | ✓      | ✗        | ✗    | ✗      | ~    |
| High-level API for new PDEs  | ✓      | ~        | ✓    | ✓      | ~    |

Table: Positioning of `manapy` among widely used open-source PDE/CFD frameworks
(FV = finite volume, FR = flux reconstruction; ✓ full, ~ partial, ✗ none).
`manapy`'s niche is a Python finite-volume framework that combines general
unstructured/hybrid meshes with a single solver implementation that runs on both
distributed CPUs and GPUs.

# Functionality

A typical workflow reads or generates a mesh, instantiates a model, sets boundary
conditions and initial data, and advances the solution while writing VTK output
for visualization. The `manapy/examples` directory contains runnable 2D and 3D
cases for each shipped solver, and `meshes/geo` provides gmsh `.geo` sources for
the example meshes (unstructured triangular/tetrahedral, structured
quadrilateral/hexahedral, hybrid, and periodic). The public API is exercised by
the `*_api.py` examples, while lower-level solvers expose the finite-volume
operators directly for method development.

# AI usage disclosure

Generative-AI tools were used to assist with debugging during the development of
the software. All AI-assisted outputs were reviewed, tested, and validated by the
authors, who take full responsibility for the code and its correctness.

# Acknowledgements

<!-- TODO: funding sources, grant numbers, contributors, and institutional support. -->

# References
