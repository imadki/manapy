# Install `petsc4py` for `manapy`

`petsc4py` is optional in `manapy`, but it is required if you want to use the PETSc-based linear solvers such as `PETScKrylovSolver` in `manapy/solvers/ls`.

This guide covers three practical installation paths:

1. `conda-forge` for the simplest setup.
2. `pip` with system MPI packages.
3. Building PETSc from source when you need a custom PETSc build.

## Prerequisites

- Python 3.8 or newer
- A working MPI toolchain
- `mpi4py` installed in the same Python environment

`manapy` already depends on `mpi4py`, but `petsc4py` is not part of the base package requirements.

## Option 1: Install from `conda-forge`

This is the shortest and most reliable path if you use Conda or Mamba.

```bash
conda install -c conda-forge openmpi mpi4py petsc petsc4py
```

After installation, verify that PETSc is visible from Python:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
```

## Option 2: Install with `pip`

You need MPI headers and compilers available before installing PETSc and `petsc4py`.

### Ubuntu / Debian

```bash
sudo apt install build-essential openmpi-bin openmpi-common libopenmpi-dev
```

### RHEL / CentOS / Rocky / AlmaLinux

```bash
sudo yum install -y openmpi openmpi-devel
```

If OpenMPI is installed in a non-default prefix, make sure `mpicc` is on `PATH`:

```bash
mpicc --version
```

Then install PETSc and `petsc4py` into the active Python environment:

```bash
export PETSC_CONFIGURE_OPTIONS="--with-debugging=0 --download-fblaslapack --with-shared-libraries=1"
python -m pip install petsc petsc4py
```

Verify the result:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

## Option 3: Build PETSc from source

Use this when you need more control over PETSc configuration, MPI, BLAS/LAPACK, or optional solver backends.

Clone PETSc:

```bash
git clone -b release https://gitlab.com/petsc/petsc.git
cd petsc
```

Configure and build:

```bash
./configure \
  --with-debugging=0 \
  --download-fblaslapack \
  --with-shared-libraries=1

make
make check
```

Export the PETSc environment variables expected by `petsc4py`:

```bash
export PETSC_DIR="$PWD"
export PETSC_ARCH="arch-linux-c-opt"
```

If you want these variables to persist across shell sessions:

```bash
echo 'export PETSC_DIR="$PETSC_DIR"' >> ~/.bashrc
echo 'export PETSC_ARCH="$PETSC_ARCH"' >> ~/.bashrc
```

Install `petsc4py` against this PETSc build:

```bash
python -m pip install src/binding/petsc4py
```

Verify the installation:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
```

## Optional: manylinux container build

This is useful if you want to build inside a controlled Linux environment.

```bash
sudo docker pull quay.io/pypa/manylinux2014_x86_64
sudo docker run -it quay.io/pypa/manylinux2014_x86_64:latest
```

Inside the container:

```bash
yum install -y openmpi openmpi-devel
export PATH="$PATH:/usr/lib64/openmpi/bin"
export PATH="$PATH:/opt/python/cp314-cp314/bin"
export PETSC_CONFIGURE_OPTIONS="--with-debugging=0 --download-fblaslapack --with-shared-libraries=1"
python -m pip install petsc petsc4py
```
## More about configuring PETSc

> https://petsc.org/release/install/install/

## More about installing PETSc

> https://petsc.org/release/install/

## Check that MPI is usable

Before debugging a `petsc4py` install, confirm the MPI compiler is available:

```bash
printf 'int main(void){return 0;}\n' > test.c
mpicc test.c -o test
```

If this fails, fix the MPI installation first.

## Notes

- `conda deactivate` can help if Conda-provided MPI or PETSc libraries are conflicting with a system or `pip` installation.
- Do not mix Conda MPI, system MPI, and custom PETSc builds unless you are doing it intentionally and know which runtime libraries are being loaded.
