# manapy

`manapy` is a Python framework for solving systems of conservation laws with the
cell-centered finite-volume method on unstructured, possibly hybrid, meshes in 2D
and 3D. The same solver code runs serially, across many MPI ranks, and on CUDA
GPUs, and a small high-level API lets you set up a simulation in a few lines.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
models
examples
api
contributing
```

## Highlights

- Finite-volume operators on unstructured / hybrid / periodic 2D and 3D meshes.
- Unified CPU (Numba) and GPU (CUDA) backends — kernels written once.
- Distributed MPI domain decomposition; linear solvers via PETSc, MUMPS, Ginkgo,
  or SciPy.
- Built-in solvers: advection, advection–diffusion, Burgers, diffusion / Poisson
  / Darcy, compressible Euler, shallow water, and shallow-water MHD.

## Indices

- {ref}`genindex`
- {ref}`modindex`
