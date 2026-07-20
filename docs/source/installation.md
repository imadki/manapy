# Installation

## Requirements

- Python 3.8+
- A working MPI runtime for `mpi4py`
- (optional) a CUDA toolchain for the GPU backend
- (optional) PETSc / MUMPS for the corresponding distributed linear solvers

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

## Optional backends

- MUMPS: see [`tools/install_mumps4py.md`](https://github.com/imadki/manapy/blob/main/tools/install_mumps4py.md)
- PETSc: see [`tools/install_petsc4py.md`](https://github.com/imadki/manapy/blob/main/tools/install_petsc4py.md)

## Verify the installation

```bash
python3 -m pytest tests                 # serial test suite
mpirun -n 4 python3 -m pytest tests     # parallel paths
```
