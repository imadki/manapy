`manapy` is a Python package for finite-volume methods on unstructured meshes.

## Requirements

- Python 3.8+
- A working MPI runtime for `mpi4py`

## Install

Standard install:

```bash
python3 -m pip install .
```

Editable install for development:

```bash
python3 -m pip install -e .
```

## Build

Build distributions locally:

```bash
python3 -m build

# 2. Install the generated wheel
pip install dist/*.whl
```


See the [c_api readme](./manapy/c_api/README.md)
- **Section 7** – *Uploading the package to PyPI*  
- **Section 4** – *Manylinux wheel build (Linux platform)* 


## Tests

```bash
python3 -m pytest tests
```

## Backend solver for Mumps and Petsc

- [Mumps](./tools/install_mumps4py.md)
- [PETSc](./tools/install_petsc4py.md)

