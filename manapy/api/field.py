import numpy as np

from manapy.ast import Variable


_BC_LOCS_2D = ("in", "out", "upper", "bottom")
_BC_LOCS_3D = ("in", "out", "upper", "bottom", "front", "back")


def _to_lambda(val):
    """Wrap a scalar or array into a (x, y, z) -> array callable."""
    if callable(val):
        return val
    return lambda x, y, z, _v=val: np.full(np.asarray(x).shape, float(_v))


class Field:
    """
    A scalar field on a Mesh with optional boundary conditions.

    Parameters
    ----------
    mesh : Mesh
    name : str
    bc   : dict, optional
        Boundary conditions per location.
        Format: {"in": ("dirichlet", value), "out": ("neumann", 0.)}
        Value can be a float or a callable f(x, y, z).
        Locations not listed default to homogeneous Neumann (zero flux).
    init : float, array, or callable f(x, y, z)
        Initial cell values (default 0.0).

    Example
    -------
    phi = Field(mesh, name="phi",
                bc={"in": ("dirichlet", 1.0),
                    "out": ("dirichlet", 0.0)},
                init=0.0)
    """

    def __init__(self, mesh, name="", bc=None, init=0.0):
        domain = mesh.domain
        self._mesh = mesh
        self._name = name

        if bc is not None:
            bc_types  = {loc: spec[0] for loc, spec in bc.items()}
            bc_values = {loc: _to_lambda(spec[1]) for loc, spec in bc.items()}
            self._var = Variable(domain=domain, name=name,
                                 BC=bc_types, values=bc_values)
        else:
            self._var = Variable(domain=domain, name=name)

        # Apply initial condition to cell centres
        c = domain.cells.center
        if callable(init):
            self._var.cell[:] = init(c[:, 0], c[:, 1], c[:, 2])
        elif isinstance(init, np.ndarray):
            self._var.cell[:] = init
        else:
            self._var.cell[:] = float(init)

        self._var.update_halo_value()
        self._var.update_ghost_value()

    # ------------------------------------------------------------------
    # Direct array access (read/write)
    # ------------------------------------------------------------------
    @property
    def cell(self):
        return self._var.cell

    @property
    def face(self):
        return self._var.face

    @property
    def node(self):
        return self._var.node

    @property
    def name(self):
        return self._name

    @property
    def var(self):
        """Underlying Variable — for use with existing solvers."""
        return self._var

    @property
    def mesh(self):
        return self._mesh
