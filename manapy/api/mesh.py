import os

from manapy.partitions import MeshPartition
from manapy.ddm import Domain
from manapy.ast import Variable
from manapy.base.base import Struct


class Mesh:
    """
    Wraps MeshPartition + Domain into a single object.

    Parameters
    ----------
    filename  : str   — path to the .msh file
    dim       : int   — 2 or 3
    backend   : str   — "numba" (default) or "python"
    cache     : bool  — cache compiled Numba functions (default True)
    precision : str   — "double" (default) or "single"
    periodic  : list  — [px, py, pz] periodicity flags (default [0,0,0])
    work_dir  : str   — directory for meshesNPROC/ partition files
                        (default: current working directory)

    Example
    -------
    mesh = Mesh("rectangle.msh", dim=2)
    """

    def __init__(self, filename, dim=2, backend="numba",
                 cache=True, precision="double",
                 periodic=None, work_dir=None):

        if periodic is None:
            periodic = [0, 0, 0]

        self._conf = Struct(
            backend=backend,
            signature=True,
            cache=cache,
            float_precision=precision,
        )

        Variable.is_called = False

        original_cwd = os.getcwd()
        if work_dir is not None:
            os.makedirs(work_dir, exist_ok=True)
            os.chdir(work_dir)

        try:
            MeshPartition(filename, dim=dim, conf=self._conf, periodic=periodic)
            self._domain = Domain(dim=dim, conf=self._conf)
        finally:
            os.chdir(original_cwd)

    @property
    def domain(self):
        return self._domain

    @property
    def conf(self):
        return self._conf

    @property
    def dim(self):
        return self._domain.dim
