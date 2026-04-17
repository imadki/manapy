# Legacy build
# You need to install Gklib and Metis first
# This installer is old and only work if you already install GKlib and Metis, use pyproject.toml installer instead.
# setup.py – build script for the manapy_domain C‑extension
# -------------------------------------------------------------
# Usage:
#      python -m pip install .
#  or  python setup.py build_ext --inplace
#
# Environment variables you can override:
#   METIS_INCLUDE_DIR   (default  ~/local/include)
#   METIS_LIB_DIR       (default  ~/local/lib)
# -------------------------------------------------------------

from setuptools import setup, Extension
import numpy as np
import os

home_folder = os.path.expanduser("~")

# Allow users to point to non‑standard METIS paths via env‑vars
METIS_INCLUDE = os.getenv("METIS_INCLUDE_DIR", f"{home_folder}/local/include")
METIS_LIB_DIR = os.getenv("METIS_LIB_DIR", f"{home_folder}/local/lib")
SOURCE = ["src/py_manapy_part.cpp", "src/partitioning.cpp", "src/utils.cpp", "src/LocalDomainStruct.cpp", "src/compute_cell_center_volume.cpp"]

# Only allow manapy_part32_32, manapy_part32_64 if you compile and link the int32 metis version
# Only allow manapy_part64_32, manapy_part64_64 if you compile and link the int64 metis version
libs = {
    "manapy_part32_32": {
        "macros": [("FLOAT_TYPE", "float"), ("INT_TYPE", "int32"), ("MODULE_NAME", "manapy_part32_32")]
    },
    "manapy_part32_64": {
        "macros": [("FLOAT_TYPE", "double"), ("INT_TYPE", "int32"), ("MODULE_NAME", "manapy_part32_64")]
    },
    # "manapy_part64_32": {
    #     "macros": [("FLOAT_TYPE", "float"), ("INT_TYPE", "int64"), ("MODULE_NAME", "manapy_part64_32")]
    # },
    # "manapy_part64_64": {
    #     "macros": [("FLOAT_TYPE", "double"), ("INT_TYPE", "int64"), ("MODULE_NAME", "manapy_part64_64")]
    # }
}
ext_modules = []
for lib_name in libs:
    ext_modules.append(
        Extension(
            name=lib_name,
            sources=SOURCE,
            include_dirs=[np.get_include(), METIS_INCLUDE, "includes"],
            library_dirs=[METIS_LIB_DIR],
            libraries=["metis", "GKlib"],
            extra_compile_args=["-O3"],
            define_macros=libs[lib_name]["macros"],
            # extra_compile_args=["-O0", "-g", "-fsanitize=address"],
            # extra_compile_args=["-O0", "-g"],
            # extra_link_args=["-fsanitize=address"],
            language="c++"
        )
    )

setup(
    name="manapy_part",
    version="1.0.1",
    description="Manapy C API",
    ext_modules=ext_modules,
    python_requires=">=3.8",
    classifiers=[],
    zip_safe=False,
)

"""
python -m pip install numpy           # ensure NumPy headers

# Need Cmake to build

# Install GKlib
git clone https://github.com/KarypisLab/GKlib.git
cd GKlib
make config cc=gcc prefix=~/local
make install
cd ..

# Install METIS
git clone https://github.com/KarypisLab/METIS.git
cd METIS
make config cc=gcc prefix=~/local
make install
cd ..

# Install manapy_domain lib
python3 -m pip install .

#gcc -O2 -shared -fPIC \
#    $(python3 -m pybind11 --includes) \  # or `python3 -m numpy --cflags`
#    -I/usr/include/metis -lmetis \
#    manapy_domain.c \
#    -o manapy_domain$(python3-config --extension-suffix)
"""