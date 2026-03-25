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
    def rectangle(cls, Lx=1.0, Ly=1.0, nx=20, ny=20,
                  cell_type="triangle", **kwargs):
        """
        Structured rectangle [0,Lx] × [0,Ly].

        Parameters
        ----------
        Lx, Ly    : float — dimensions (default 1.0)
        nx, ny    : int   — cells per direction (default 20)
        cell_type : str   — "triangle" (default) or "quad"
        **kwargs          — forwarded to Mesh() (backend, cache, precision, …)

        Examples
        --------
        mesh = Mesh.rectangle(Lx=2.0, Ly=1.0, nx=40, ny=20)
        mesh = Mesh.rectangle(Lx=1.0, Ly=1.0, nx=20, ny=20, cell_type="quad")
        """
        from manapy.api.meshgen import rectangle as _gen
        path = _gen(Lx=Lx, Ly=Ly, nx=nx, ny=ny, cell_type=cell_type)
        return cls(path, dim=2, **kwargs)

    @classmethod
    def square(cls, L=1.0, n=20, cell_type="triangle", **kwargs):
        """
        Structured square [0,L] × [0,L].

        Parameters
        ----------
        cell_type : "triangle" (default) or "quad"

        Examples
        --------
        mesh = Mesh.square(n=30)
        mesh = Mesh.square(n=30, cell_type="quad")
        """
        from manapy.api.meshgen import square as _gen
        path = _gen(L=L, n=n, cell_type=cell_type)
        return cls(path, dim=2, **kwargs)

    @classmethod
    def cube(cls, L=1.0, n=8, cell_type="tetra", **kwargs):
        """
        Structured cube [0,L]³.

        Parameters
        ----------
        cell_type : "tetra" (default) or "hex"

        Examples
        --------
        mesh = Mesh.cube(n=10)
        mesh = Mesh.cube(n=8, cell_type="hex")
        """
        from manapy.api.meshgen import cube as _gen
        path = _gen(L=L, n=n, cell_type=cell_type)
        return cls(path, dim=3, **kwargs)

    @classmethod
    def box(cls, Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8,
            cell_type="tetra", **kwargs):
        """
        Structured box [0,Lx] × [0,Ly] × [0,Lz].

        Parameters
        ----------
        cell_type : "tetra" (default) or "hex"

        Examples
        --------
        mesh = Mesh.box(Lx=2.0, Ly=1.0, Lz=1.0, nx=20, ny=10, nz=10)
        mesh = Mesh.box(nx=10, ny=10, nz=10, cell_type="hex")
        """
        from manapy.api.meshgen import box as _gen
        path = _gen(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz,
                    cell_type=cell_type)
        return cls(path, dim=3, **kwargs)
