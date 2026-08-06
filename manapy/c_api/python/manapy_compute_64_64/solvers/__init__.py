"""Solver kernels compiled for float64 data and int64 indices.

Submodules (each its own compiled extension):
    advec     -- explicit advection solver kernels (CPU + CUDA)
    advecdiff -- advection-diffusion solver kernels (CPU + CUDA)
    diffusion -- pure-diffusion solver kernels (CPU + CUDA)
    utils     -- kernels common to all solvers, e.g. initial conditions and
                 the forward-Euler update (mixed CPU + CUDA)
"""

# Each solver is its own compiled extension, loaded lazily (PEP 562): touching
# `solvers.diffusion` imports it on demand, so importing `solvers` alone costs
# nothing.
import importlib as _importlib

_SUBMODULES = ("advec", "advecdiff", "diffusion", "utils")


def __getattr__(name):
    if name in _SUBMODULES:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBMODULES])
