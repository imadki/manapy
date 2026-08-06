from typing import TYPE_CHECKING, Sequence
from manapy.backends.ManapyArray import ManapyArray
from manapy.compute import BoundaryCompute
from manapy.domain import Domain

if TYPE_CHECKING:
  from manapy.core.Variable import Variable



class Boundary:
  _VALID_LOCS = ("in", "out", "bottom", "upper", "front", "back")

  # Vector (component-coupled) BC types. These cannot be applied per scalar
  # field; they are handled by update_slip_ghost(velocity) instead, so their
  # per-scalar func_ghost stays None and update_ghost_value skips them.
  _VECTOR_TYPES = ("slip",)

  _VALID_TYPES = (
    "dirichlet",
    "neumann",
    "neumannNH",
    "nonslip",
    "periodic",
    "slip",
  )

  def __init__(
      self,
      BCtype: str,
      BCloc: str,
      BCtypeindex: int,
      domain: Domain,
      default_value_face: ManapyArray,
      default_value_node: ManapyArray,
      default_value_halo: ManapyArray,
  ) -> None:

    if domain is None:
      raise ValueError("domain must be given")
    if not isinstance(BCtypeindex, int):
      raise ValueError("BCtypeindex must be an integer")
    if BCloc not in Boundary._VALID_LOCS:
      raise ValueError(f"unknown BCloc: {BCloc}")
    if BCtype not in Boundary._VALID_TYPES:
      raise ValueError(f"unknown BCtype: {BCtype}")

    self._BCtype = BCtype
    self.BCvalueface = default_value_face
    self.BCvaluenode = default_value_node
    self.BCvaluehalo = default_value_halo
    self._domain = domain
    self.config = domain.config

    # Resolves the (device, dim, BCtype) kernels
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

    # Keep the location-to-domain-table decision visible. These tables are
    # already ManapyArray instances, converted once by Domain.__init__.
    if BCloc == "in":
      self._BCfaces = self._domain.infaces
      self._BCnodes = self._domain.innodes
    elif BCloc == "out":
      self._BCfaces = self._domain.outfaces
      self._BCnodes = self._domain.outnodes
    elif BCloc == "bottom":
      self._BCfaces = self._domain.bottomfaces
      self._BCnodes = self._domain.bottomnodes
    elif BCloc == "upper":
      self._BCfaces = self._domain.upperfaces
      self._BCnodes = self._domain.uppernodes
    elif BCloc == "front":
      self._BCfaces = self._domain.frontfaces
      self._BCnodes = self._domain.frontnodes
    else:
      self._BCfaces = self._domain.backfaces
      self._BCnodes = self._domain.backnodes
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
  def domain(self) -> Domain:
    return self._domain

  @property
  def BCfaces(self) -> ManapyArray:
    return self._BCfaces

  @property
  def BCtype(self) -> str:
    return self._BCtype

  @property
  def BCnodes(self) -> ManapyArray:
    return self._BCnodes

  @property
  def BCtypeindex(self) -> int:
    return self._BCtypeindex

  @staticmethod
  def update_slip_ghost(velocity: Sequence[Variable]) -> None:
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
