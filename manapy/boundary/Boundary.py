from manapy.backends.ManapyArray import ManapyArray
from manapy.compute import BoundaryCompute
from manapy.domain import Domain

class Boundary:
  # Valid boundary patches. The matching domain arrays follow the naming
  # convention <loc>faces / <loc>nodes (e.g. infaces/innodes, outfaces/...).
  _VALID_LOCS = ("in", "out", "bottom", "upper", "front", "back")

  # Vector (component-coupled) BC types. These cannot be applied per scalar
  # field; they are handled by update_slip_ghost(velocity) instead, so their
  # per-scalar func_ghost stays None and update_ghost_value skips them.
  _VECTOR_TYPES = ("slip",)

  def __init__(self, BCtype:str, BCvalueface:'float[:]', BCvaluenode:'float[:]', BCvaluehalo:'float[:]',
               BCloc:str, BCtypeindex:int, domain:Domain):

    if domain is None:
      raise ValueError("domain must be given")
    if not isinstance(BCtypeindex, int):
      raise ValueError("BCtypeindex must be an integer")
    if BCloc not in Boundary._VALID_LOCS:
      raise ValueError(f"unknown BCloc: {BCloc}")

    self._BCtype = BCtype
    self.BCvalueface = BCvalueface
    self.BCvaluenode = BCvaluenode
    self.BCvaluehalo = BCvaluehalo
    self._domain = domain
    self.config = domain.config

    # Resolves the (device, dim, BCtype) kernel pair once, here. Raises
    # ValueError on an unknown BCtype -- the only validation of the type, so it
    # has to happen before anything else relies on it.
    self.compute = BoundaryCompute(self.config, domain.dim, BCtype)
    #: True for a component-coupled BC (slip): see func_ghost below.
    self.is_vector = BCtype in Boundary._VECTOR_TYPES

    # neumannNH gradient constants: constNH is per boundary face (used by the
    # ghost kernel), constNHGhost per halo ghost (used by the haloghost kernel).
    # Both are replaced by the real arrays in Variable._update_boundaries.
    # Allocated on the config device, so the kernels never see a bare ndarray:
    # the other BC kinds do not read them, but they still travel to the kernel.
    self.constNH = ManapyArray.zeros(1, self.config.float_dtype, self.config.device)
    self.constNHGhost = ManapyArray.zeros(1, self.config.float_dtype, self.config.device)

    # Domain tables: already ManapyArray, converted once by Domain.__init__.
    self._BCfaces = getattr(self._domain, BCloc + "faces")
    self._BCnodes = getattr(self._domain, BCloc + "nodes")
    self._BCtypeindex = BCtypeindex

    if self.is_vector:
      # Coupled (vector) BC: the kernel takes the whole velocity group at once,
      # so it has no per-scalar signature. Leaving func_ghost None is what makes
      # Variable.update_ghost_value skip this patch; update_slip_ghost applies
      # it instead, through slip_ghost / slip_haloghost.
      self.func_ghost = None
      self.func_haloghost = None
      self.slip_ghost = self.compute.ghost
      self.slip_haloghost = self.compute.haloghost
    else:
      self.func_ghost = self.compute.ghost
      self.func_haloghost = self.compute.haloghost
      self.slip_ghost = None
      self.slip_haloghost = None


  @property
  def domain(self):
    return self._domain

  @property
  def BCfaces(self):
    return self._BCfaces

  @property
  def BCtype(self) -> str:
    return self._BCtype

  @property
  def BCnodes(self):
    return self._BCnodes

  @property
  def BCtypeindex(self):
    return self._BCtypeindex


def update_slip_ghost(velocity):
  """Apply free-slip ghost (and haloghost) values to a velocity field.

  `velocity` is the tuple of velocity-component Variables: ``(u, v)`` in 2D or
  ``(u, v, w)`` in 3D. Every boundary patch tagged ``"slip"`` on the first
  component is treated as a free-slip wall (normal velocity reflected,
  tangential preserved). Call this in place of the per-component
  ``update_ghost_value()`` on the slip walls; the other BC types remain handled
  by ``update_ghost_value()``.

  Each slip patch already carries its own dimension-resolved kernel pair (its
  Boundary was built with the domain's dim), so no dispatch is left to do here.
  """
  u = velocity[0]
  domain = u.domain
  dim = domain.dim

  if dim == 2:
    v = velocity[1]
    for bc in u.BCs.values():
      if bc is None or bc.BCtype != "slip":
        continue
      bc.slip_ghost(u.cell, v.cell, u.ghost, v.ghost,
                    domain.faces.cellid, bc.BCfaces, domain.faces.normal)
      bc.slip_haloghost(u.halo, v.halo, u.haloghost, v.haloghost,
                        domain.nodes.haloghostid, domain.ghost.ext_info_int,
                        domain.ghost.ext_info_flt, bc.BCtypeindex, domain.halonodes)
  else:
    v = velocity[1]
    w = velocity[2]
    for bc in u.BCs.values():
      if bc is None or bc.BCtype != "slip":
        continue
      bc.slip_ghost(u.cell, v.cell, w.cell, u.ghost, v.ghost, w.ghost,
                    domain.faces.cellid, bc.BCfaces, domain.faces.normal)
      bc.slip_haloghost(u.halo, v.halo, w.halo, u.haloghost, v.haloghost, w.haloghost,
                        domain.nodes.haloghostid, domain.ghost.ext_info_int,
                        domain.ghost.ext_info_flt, bc.BCtypeindex, domain.halonodes)
