# Examples

Runnable 2D and 3D cases for every solver live in
[`manapy/examples`](https://github.com/imadki/manapy/tree/main/manapy/examples).
Files ending in `_api.py` use the high-level API; the others drive the solvers
directly.

## Meshes

Mesh sources (gmsh `.geo`) are in
[`meshes/geo`](https://github.com/imadki/manapy/tree/main/meshes/geo). Generate a
mesh before running an example that reads one from disk, e.g.:

```bash
gmsh -2 meshes/geo/uns_square.geo -o meshes/geo/uns_square.msh   # 2D
gmsh -3 meshes/geo/uns_cube.geo   -o meshes/geo/uns_cube.msh     # 3D
```

The naming convention and the mesh/example mapping are documented in
[`meshes/geo/README.md`](https://github.com/imadki/manapy/blob/main/meshes/geo/README.md).

## Running in parallel and on the GPU

Every example runs unchanged under MPI:

```bash
mpirun -n 4 python3 manapy/examples/2D/advection/advection2d_api.py
```

The `*_gpu.py` examples target the CUDA backend.
