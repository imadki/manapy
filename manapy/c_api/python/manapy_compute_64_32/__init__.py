"""Kernels compiled for float64 data and int32 indices.

Grouped into submodules:
    core     -- 2D/3D cell-gradient and limiter kernels (CPU + CUDA)
    boundary -- ghost / halo-ghost boundary-condition kernels (CPU + CUDA)
    domain   -- mesh connectivity/geometry kernels (CPU-only)
    partitioning -- METIS-backed domain decomposition (CPU-only)
    solvers  -- PDE solver kernels, itself split into submodules:
                 advec     -- explicit advection solver (CPU + CUDA)
                 advecdiff -- advection-diffusion solver (CPU + CUDA)
                 diffusion -- pure-diffusion solver (CPU + CUDA)
                 utils     -- kernels common to all solvers (mixed CPU + CUDA)

Import whichever you need, e.g. `from manapy_compute_64_32 import core`
or `from manapy_compute_64_32.solvers import advec`.
"""

# Submodules are separate compiled extensions, so they are loaded lazily (PEP
# 562): `manapy_compute_64_32.domain` works without an explicit
# `import manapy_compute_64_32.domain` first, while nothing dlopens its .so
# -- nor pulls in libcudart -- until the attribute is actually touched.
import importlib as _importlib

_SUBMODULES = ("boundary", "core", "domain", "partitioning", "solvers")


def __getattr__(name):
    if name in _SUBMODULES:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBMODULES])
