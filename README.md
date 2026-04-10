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

Build distributions locally:

```bash
python3 -m build
```

## Tests

```bash
python3 -m pytest tests
```

## Backend solver for Mumps and Petsc

MUMPS support:

```bash
sudo apt install libmumps-ptscotch-dev
python3 -m pip install mumps4py
```

PETSc support:

```bash
python3 -m pip install petsc4py
```

Follow the PETSc installation guide if your system packages are not already available: https://petsc.org/release/install/
