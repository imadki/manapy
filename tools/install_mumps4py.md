# Install `mumps4py` for `manapy`

`mumps4py` is optional in `manapy`, but it is required if you want to use `MUMPSSolver` from `manapy/solvers/ls`.

This document explains what [install_mumps4py.sh](/media/aben-ham/SSD/aben-ham/work/manapy/tools/install_mumps4py.sh) does.

## What the script does

The script is a convenience installer for Debian-based systems. It avoids a full system-wide MUMPS installation by:

1. Downloading the `libmumps-dev` Debian package with `apt download`.
2. Extracting the package with `dpkg -x` into a temporary directory.
3. Detecting the matching MUMPS library directory from the extracted files.
4. Downloading the matching upstream MUMPS source archive to obtain headers.
5. Exporting the build variables expected by `mumps4py`:
   - `MUMPS_INC`
   - `MUMPS_LIB`
   - `MUMPS_SOLVERS`
6. Installing `mumps4py` with `python3 -m pip install`.

If the script is executed from inside the `manapy` repository, it installs from the local repository root. Otherwise, it clones `https://github.com/imadki/mumps4py.git` and installs from that checkout.

## Usage

Run the installer from the repository root:

```bash
bash tools/install_mumps4py.sh
```

## Environment variables used by the build

The script exports these variables before installing `mumps4py`:

```bash
MUMPS_INC=<temporary-path>/MUMPS_<version>/include
MUMPS_LIB=<temporary-path>/dist/.../lib
MUMPS_SOLVERS=dmumps,cmumps,zmumps,smumps
```

These are printed during execution so you can inspect what was detected.

## Expected workflow

On success, the script should:

1. Create a temporary directory named like `mumps-installer-XXXXXX`.
2. Download `libmumps-dev`.
3. Extract the package and locate MUMPS shared libraries.
4. Download the matching `MUMPS_<version>.tar.gz` archive.
5. Install `mumps4py` with `pip`.

## Verify the installation

First, verify that Python can import `mumps4py`:

```bash
python3 -c "import mumps4py.mumps_solver as mumps; print(mumps.__name__)"
```
