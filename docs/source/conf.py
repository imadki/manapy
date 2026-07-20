"""Sphinx configuration for the manapy documentation."""
import os
import sys

# Make the package importable for autodoc (repo root is three levels up).
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "manapy"
author = "Manapy Developers"
copyright = "2019-2025, Manapy Developers"

# Read the version without importing the package (which pulls in heavy deps).
version = "1.0.0"
try:
    _ns = {}
    with open(os.path.abspath("../../manapy/version.py")) as _f:
        exec(_f.read(), _ns)
    version = _ns.get("__version__", version)
except Exception:
    pass
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# manapy imports compiled extensions and MPI/GPU runtimes that are not available
# on a docs builder; mock them so autodoc can import the pure-Python API.
autodoc_mock_imports = [
    "mpi4py",
    "numba",
    "numpy",
    "scipy",
    "meshio",
    "petsc4py",
    "mumps4py",
    "pyccel",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"manapy {release}"
