# setup.py – build script for the manapy_domain C‑extension
# -------------------------------------------------------------
# Usage:
#      python -m pip install .            # build & install in‑place
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


ext_modules = [
    Extension(
        name="manapy_part32_32",
        sources=SOURCE,
        include_dirs=[np.get_include(), METIS_INCLUDE, "includes"],
        library_dirs=[METIS_LIB_DIR],
        libraries=["metis", "GKlib"],
        extra_compile_args=["-O3"],
        define_macros=[
            ("FLOAT_TYPE", "float"),
            ("INT_TYPE", "int32"),
            ("MODULE_NAME", "manapy_part32_32")
        ],
        # extra_compile_args=["-O0", "-g", "-fsanitize=address"],
        # extra_compile_args=["-O0", "-g"],
        # extra_link_args=["-fsanitize=address"],
        language="c++"
    ),

    Extension(
        name="manapy_part32_64",
        sources=SOURCE,
        include_dirs=[np.get_include(), METIS_INCLUDE, "includes"],
        library_dirs=[METIS_LIB_DIR],
        libraries=["metis", "GKlib"],
        extra_compile_args=["-O3"],
        define_macros=[
            ("FLOAT_TYPE", "double"),
            ("INT_TYPE", "int32"),
            ("MODULE_NAME", "manapy_part32_64")
        ],
        # LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3 program.py
        # extra_compile_args=["-O0", "-g", "-fsanitize=address"],
        # extra_compile_args=["-O0", "-g"],
        # extra_link_args=["-fsanitize=address"],
        language="c++"
    )
]

setup(
    name="manapy_domain",
    version="0.1.0",
    description="Manapy C API",
    ext_modules=ext_modules,
    python_requires=">=3.8",
    classifiers=[],
    zip_safe=False,
)
