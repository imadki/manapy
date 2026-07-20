import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.api import meshgen


def _detect_dim(filename):
  """Best-effort dimension detection from the mesh file (defaults to 2)."""
  try:
    import meshio
    m = meshio.read(filename)
    types = {c.type for c in m.cells}
    if types & {"tetra", "hexahedron", "wedge", "pyramid"}:
      return 3
  except Exception:
    pass
  return 2


def _apply_init(var, domain, init):
  """Set a Variable's cell values from a constant, a callable f(x,y,z), or an array."""
  if init is None:
    return
  if callable(init):
    c = domain.cells.center
    var.cell[:] = init(c[:, 0], c[:, 1], c[:, 2])
  elif np.isscalar(init):
    var.cell[:] = float(init)
  else:
    var.cell[:] = np.asarray(init)


def _parse_bc(bc):
  """Split a user bc dict into (types, values) for Variable.

  Each entry maps a patch to either a type string ("neumann", "slip", ...) or a
  (type, value) pair for value-carrying types ("dirichlet", "neumannNH").
  """
  if bc is None:
    return None, None
  types, values = {}, {}
  for loc, spec in bc.items():
    if isinstance(spec, (tuple, list)):
      types[loc] = spec[0]
      values[loc] = spec[1]
    else:
      types[loc] = spec
  return types, (values or None)


class Mesh:
  """High-level mesh wrapper around Domain + a Variable factory.

  Parameters
  ----------
  filename     : path to the mesh file (.msh)
  dim          : 2 or 3; auto-detected from the file when None
  backend      : passed to Domain.create_domain (None = manapy default)
  partitioning : a Partitioning.* method (default Par_Nodal)
  recreate     : rebuild the local partition files (default True)

  Examples
  --------
  mesh = Mesh("carre.msh")                 # dim auto-detected
  mesh = Mesh.rectangle(n=(128, 128))      # generated on the fly, no .msh file
  P = mesh.field("P", init=lambda x, y, z: 1 - x,
                 bc={"in": ("dirichlet", 1), "out": ("dirichlet", 0),
                     "upper": "neumann", "bottom": "neumann"})
  """

  def __init__(self, filename, dim=None, backend=None,
               partitioning=Partitioning.Par_Nodal, recreate=True):
    if dim is None:
      dim = _detect_dim(filename)
    self._domain = Domain.create_domain(filename, dim, partitioning,
                                        recreate=recreate, backend=backend)
    self._filename = filename

  # ------------------------------------------------------------------ props
  @property
  def domain(self):
    return self._domain

  @property
  def dim(self):
    return self._domain.dim

  # ----------------------------------------------------------- field factory
  def field(self, name=None, init=None, bc=None, limiter=None):
    """Create a Variable on this mesh.

    init : constant, callable f(x, y, z), or array → initial cell values.
    bc   : dict {patch: type | (type, value)} → boundary conditions.
    """
    types, values = _parse_bc(bc)
    var = Variable(domain=self._domain, BC=types, values_dict=values, name=name, limiter=limiter)
    _apply_init(var, self._domain, init)
    return var

  # ------------------------------------------------- on-the-fly generators
  @classmethod
  def rectangle(cls, bounds=((0, 1), (0, 1)), n=20, cell_type="triangle",
                transfinite=True, recombine=None, **kwargs):
    """Generate a 2D rectangle mesh (no .msh file needed).

    transfinite=False -> unstructured; recombine=True -> quads (or hybrid if also
    unstructured). See manapy.api.meshgen.rectangle for the full combination table."""
    path = meshgen.rectangle(bounds=bounds, n=n, cell_type=cell_type,
                             transfinite=transfinite, recombine=recombine)
    return cls(path, dim=2, **kwargs)

  @classmethod
  def box(cls, bounds=((0, 1), (0, 1), (0, 1)), n=10, cell_type="tetra",
          transfinite=True, recombine=None, **kwargs):
    """Generate a 3D box mesh (no .msh file needed).

    transfinite=False -> unstructured; recombine=True -> hexes (or hybrid if also
    unstructured). See manapy.api.meshgen.box for the full combination table."""
    path = meshgen.box(bounds=bounds, n=n, cell_type=cell_type,
                       transfinite=transfinite, recombine=recombine)
    return cls(path, dim=3, **kwargs)
