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

    # ------------------------------------------------------------------
    # Factory methods — generate mesh on the fly, no .msh file needed
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls, dim=2, bounds=None, n=20,
                 cell_type=None, **kwargs):
        """
        Generate a structured mesh on the fly — no .msh file needed.

        Parameters
        ----------
        dim       : int — 2 or 3
        bounds    : list of (min, max) per axis.
                    Default: ((0,1),(0,1)) in 2D, ((0,1),(0,1),(0,1)) in 3D.
        n         : int or tuple — cells per direction.
                    int  → same value for all axes.
                    tuple → (nx, ny) in 2D, (nx, ny, nz) in 3D.
        cell_type : str, optional
                    2D: "triangle" (default) or "quad"
                    3D: "tetra"    (default) or "hex"
        **kwargs  — forwarded to Mesh() (backend, cache, precision, …)

        Examples
        --------
        # 2D unit square, triangles
        mesh = Mesh.generate(dim=2)

        # 2D rectangle with quads
        mesh = Mesh.generate(dim=2, bounds=((0,2),(0,1)), n=(40,20),
                             cell_type="quad")

        # 3D unit cube, hexahedra
        mesh = Mesh.generate(dim=3, n=10, cell_type="hex")

        # 3D box with custom bounds
        mesh = Mesh.generate(dim=3, bounds=((0,2),(0,1),(0,0.5)), n=(20,10,5))
        """
        if dim == 2:
            from manapy.api.meshgen import rectangle as _gen
            if bounds is None:
                bounds = ((0, 1), (0, 1))
            path = _gen(bounds=bounds, n=n,
                        cell_type=cell_type or "triangle")
        elif dim == 3:
            from manapy.api.meshgen import box as _gen
            if bounds is None:
                bounds = ((0, 1), (0, 1), (0, 1))
            path = _gen(bounds=bounds, n=n,
                        cell_type=cell_type or "tetra")
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        return cls(path, dim=dim, **kwargs)
