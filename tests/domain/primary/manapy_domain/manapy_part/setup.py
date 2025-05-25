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

ext_modules = [
    Extension(
        name="manapy_domain",
        sources=["create_local_domain.cpp"],
        include_dirs=[np.get_include(), METIS_INCLUDE],
        library_dirs=[METIS_LIB_DIR],
        libraries=["metis", "GKlib"],
        extra_compile_args=["-O3"],
        language="c++"
    )
]

setup(
    name="manapy_domain",
    version="0.1.0",
    description="Graph partitioning utilities based on METIS.",
    ext_modules=ext_modules,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
