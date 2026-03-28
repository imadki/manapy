# Manapy C API Setup Guide

This project supports installation via the modern `pyproject.toml` (recommended) and the legacy `setup.py` method.

## Project Structure & Dependencies
The library consists of several package variants based on integer precision:
- `manapy_part32_32`
- `manapy_part32_64`
- `manapy_part64_32`
- `manapy_part64_64`

These rely on **GKlib** and **METIS** (configured for 32-bit or 64-bit integers accordingly) as core dependencies.

---

## 1. Installation from PyPI
To install the pre-compiled package directly from PyPI (if available for your platform):
```bash
pip install manapy_part
```

## 2. Installation from Source (Modern `.toml` method)
To install from the local source directory using the `pyproject.toml` configuration (which uses `scikit-build-core` and `CMake`):

```bash
# Ensure you are in the project root containing pyproject.toml
pip install .
```

> [!IMPORTANT]
> If you have a legacy `setup.py` present, modern `pip` (>=19.0) will prefer `pyproject.toml` by default. If you suspect `setup.py` is being used, you can explicitly force the PEP 517 build (which will use `pyproject.toml`) via:
> `pip install . --use-pep517`

## 3. Standard Wheel Build & Local Installation
To build a wheel for your specific system and then install it:

```bash
# 1. Build the wheel
python3 -m build --wheel

# 2. Install the generated wheel
pip install dist/*.whl
```

## 4. Manylinux Wheel Build (Linux Platform)
To build portable Linux wheels (`manylinux`) that will work across different distributions without compilation, use `cibuildwheel`:

```bash
python3 -m cibuildwheel --platform linux
```

---

## 5. Pre-requisite Libraries for Build
Before using the build tools above, ensure you have the necessary libraries installed:

```bash
python3 -m pip install scikit-build-core cibuildwheel build
```

**Requirements for `cibuildwheel`:**
- **Python 3.11** must be installed.
- **Docker** must be installed, running, and accessible to your user (e.g., your user must be in the `docker` group) because `cibuildwheel` runs inside manylinux Docker containers. Use `sudo usermod -aG docker $USER` to add your user to the docker group. and `newgrp docker` to apply the changes.

---

## 6. Legacy Method using `setup.py`
If you need to use the old installation method via `setup.py` instead of the modern CMake + `pyproject.toml` workflow, you are required to manage dependencies manually:

1. **Manual Dependency Installation:** You must manually install `GKlib` and `METIS` (with 32/64 bit int support) on your system.
2. **Path Modification:** You will need to edit the `setup.py` file to modify the library paths (`include_dirs` and `library_dirs`) so they point to your local installations of `GKlib` and `METIS`.

---

## 7. Uploading the Package to PyPI
Once you have successfully built your wheels (they will be located in the `wheelhouse/` or `dist/` directories depending on the build method), you can upload them to the Python Package Index using `twine`:

```bash
# First, install twine
python3 -m pip install twine

# Then, upload the built wheels
python3 -m twine upload wheelhouse/*.whl
```

## Notes

### Build and Distribution Configuration

This project uses `cibuildwheel` and `scikit-build-core` to build and distribute precompiled Python wheels for Linux. The configuration is designed to maximize compatibility while ensuring reliable builds for native C/C++ extensions.

#### `pyproject.toml` Settings Reference

| Setting | Value | Description |
| :--- | :--- | :--- |
| `build` | `cp38-* … cp314-*` | Builds wheels for CPython 3.8 – 3.14 |
| `skip` | `*-musllinux*` | Excludes musl-based Linux (e.g. Alpine Linux) |
| `manylinux-x86_64-image` | `manylinux2014` | Based on CentOS 7 / glibc ≥ 2.17 |
| `cmake.build-type` | `Release` | Enables full compiler optimizations |

#### manylinux2014

Wheels are built against the `manylinux2014` standard. This ensures that binaries produced inside a controlled Docker container are compatible with a wide range of modern Linux distributions (Ubuntu, Debian, Fedora, etc.) — without users needing to compile anything themselves.

#### Skipped Platforms

`musllinux` targets (e.g. Alpine Linux) are excluded because they rely on the `musl` C library instead of `glibc`, which is not currently supported by this project.














## Requirements for Building and Installing

This project provides precompiled wheels for easy installation, but also supports building from source when needed. The requirements differ depending on whether you are a developer or a user.

---

### For End Users (Installing the Package)

If you install the package using:

```
pip install manapy_domain
```

you typically do **not need to build anything manually**.

#### Requirements

* Python (supported versions: 3.8 and above, depending on available wheels)
* pip

That’s all.
If a compatible wheel is available for your system, pip will download it and install it directly.

#### When compilation is NOT required

* You are using a standard Linux distribution (glibc-based)
* Your Python version matches one of the prebuilt wheels
* You are not using Alpine Linux (musl)

#### When compilation MAY be required

* No wheel exists for your platform or Python version
* You are using a non-standard system (e.g., Alpine Linux)
* You explicitly install from source

---

### For Developers (Building from Source)

If you want to build the project locally or contribute to development, you will need a full build environment.

#### Required tools

* Python (matching your target version)
* CMake (≥ 3.17)
* C/C++ compiler (e.g., gcc, g++)
* Git
* pip

#### Python build dependencies

These are automatically handled by `pyproject.toml`, but may include:

* scikit-build-core
* numpy
* (optionally) pybind11

---

### Building the Project

To build a wheel locally:

```
python -m build --wheel
```

This will:

* Configure the project using CMake
* Compile the C++ sources
* Produce a `.whl` file in the `dist/` directory

---

### Building Portable Linux Wheels (Recommended)

To build production-ready Linux wheels:

```
cibuildwheel --platform linux
```

#### Additional requirement

* Docker (required for manylinux builds)

This ensures:

* Compatibility across Linux distributions
* Proper bundling of native dependencies (GKlib, METIS)

---

### Notes on Native Dependencies

The project includes and builds the following libraries internally:

* GKlib
* METIS

These are compiled during the build process and do not need to be installed separately by the user.
