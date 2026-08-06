# The model classes live in `manapy.api.models`, which pulls in the solvers that
# are still on the old runtime-compiled backend (advecdiff, ls, ...). Importing
# them eagerly would make *every* `manapy.api` import fail -- including
# `from manapy.api import meshgen`, which needs none of that. So the names are
# resolved lazily (PEP 562, same idiom as the manapy_compute packages): you only
# pay for -- and only break on -- what you actually touch.
import importlib as _importlib

from manapy.api.mesh import Mesh
from manapy.api import meshgen

_MODEL_NAMES = ("AdvectionModel", "DiffusionModel", "PoissonModel", "DarcyModel")

__all__ = ["Mesh", "meshgen", *_MODEL_NAMES]


def __getattr__(name):
  if name in _MODEL_NAMES:
    return getattr(_importlib.import_module(".models", __name__), name)
  if name == "models":
    return _importlib.import_module(".models", __name__)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
  return sorted([*globals(), *_MODEL_NAMES, "models"])
