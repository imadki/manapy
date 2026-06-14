from manapy.domain import Domain
import manapy.boundary.bc_compute as bc_compute
import numpy as np
import manapy.backends.types as types
from manapy.backends.compile_fun import compile

class Boundary:
  # Cache of compiled (ghost, haloghost) kernels per BC kind. Mirrors the old
  # compile_func: only the boundary types actually used are ever compiled.
  _compiled = {}

  # Valid boundary patches. The matching domain arrays follow the naming
  # convention <loc>faces / <loc>nodes (e.g. infaces/innodes, outfaces/...).
  _VALID_LOCS = ("in", "out", "bottom", "upper", "front", "back")

  # BC type -> kernel "kind" in bc_compute.GHOST_BODIES / HALOGHOST_BODIES.
  # Add an entry here (and the matching kernel bodies) to support a new type.
  _KIND = {
    "neumann":   "neumann",
    "periodic":  "neumann",
    "dirichlet": "dirichlet",
    "neumannNH": "neumannNH",
    "nonslip":   "nonslip",
  }

  # Vector (component-coupled) BC types. These cannot be applied per scalar
  # field; they are handled by update_slip_ghost(velocity) instead, so their
  # per-scalar func_ghost stays None and update_ghost_value skips them.
  _VECTOR_TYPES = ("slip",)

  # Cache of compiled coupled slip kernels, keyed by (backend, dim).
  _slip_compiled = {}

  @classmethod
  def _get_slip_funcs(cls, dim, backend):
    key = (backend.name, dim)
    if key in cls._slip_compiled:
      return cls._slip_compiled[key]
    if dim == 2:
      ghost, haloghost = bc_compute.ghost_value_slip_2d, bc_compute.haloghost_value_slip_2d
      ghost_size, halo_size = 5, 8        # index of bc_faces / d_halonodes
    else:
      ghost, haloghost = bc_compute.ghost_value_slip_3d, bc_compute.haloghost_value_slip_3d
      ghost_size, halo_size = 8, 10
    cls._slip_compiled[key] = (backend.make_gridstride_kernel(ghost, size_arg=ghost_size),
                               backend.make_gridstride_kernel(haloghost, size_arg=halo_size))
    return cls._slip_compiled[key]

  @classmethod
  def _get_funcs(cls, kind, backend):
    key = (backend.name, kind)
    if key in cls._compiled:
      return cls._compiled[key]
    cls._compiled[key] = (backend.make_gridstride_kernel(bc_compute.GHOST_BODIES[kind], size_arg=3),
                          backend.make_gridstride_kernel(bc_compute.HALOGHOST_BODIES[kind], size_arg=6))
    return cls._compiled[key]

  def __init__(self, BCtype:str, BCvalueface:'float[:]', BCvaluenode:'float[:]', BCvaluehalo:'float[:]',
               BCloc:str, BCtypeindex:int, domain:Domain):

    if domain is None:
      raise ValueError("domain must be given")
    if not isinstance(BCtypeindex, int):
      raise ValueError("BCtypeindex must be an integer")

    self._BCtype = BCtype
    self.BCvalueface = BCvalueface
    self.BCvaluenode = BCvaluenode
    self.BCvaluehalo = BCvaluehalo
    self._domain = domain
    self.backend = domain.backend

    self._func_ghost_args = []
    self._func_haloghost_args = []

    self.constNH = np.zeros(1, dtype=types.np_float_type)
    self.constNHNode = np.zeros(1, dtype=types.np_float_type)
    if self.backend.name == "gpu":
      from manapy.backends.gpu import GPUArray
      self.constNH = GPUArray(self.constNH)
      self.constNHNode = GPUArray(self.constNHNode)

    if BCloc not in Boundary._VALID_LOCS:
      raise ValueError(f"unknown BCloc: {BCloc}")
    self._BCfaces = getattr(self._domain, BCloc + "faces")
    self._BCnodes = getattr(self._domain, BCloc + "nodes")
    self._BCtypeindex = BCtypeindex

    if self._BCtype in Boundary._VECTOR_TYPES:
      # coupled (vector) BC: no per-scalar kernel; see update_slip_ghost
      self._func_ghost = None
      self._func_haloghost = None
    else:
      kind = Boundary._KIND.get(self._BCtype)
      if kind is None:
        raise ValueError(f"unknown BCtype: {BCtype}")
      self._func_ghost, self._func_haloghost = Boundary._get_funcs(kind, self.backend)

    self.func_ghost = self._func_ghost
    self.func_haloghost = self._func_haloghost


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
  """
  u = velocity[0]
  domain = u.domain
  dim = domain.dim
  gfunc, hfunc = Boundary._get_slip_funcs(dim, domain.backend)

  if dim == 2:
    v = velocity[1]
    for bc in u.BCs.values():
      if bc is None or bc.BCtype != "slip":
        continue
      gfunc(u.cell, v.cell, u.ghost, v.ghost,
            domain.faces.cellid, bc.BCfaces, domain.faces.normal)
      hfunc(u.halo, v.halo, u.haloghost, v.haloghost,
            domain.nodes.haloghostid, domain.ghost.ext_info_int,
            domain.ghost.ext_info_flt, bc.BCtypeindex, domain.halonodes)
  else:
    v = velocity[1]
    w = velocity[2]
    for bc in u.BCs.values():
      if bc is None or bc.BCtype != "slip":
        continue
      gfunc(u.cell, v.cell, w.cell, u.ghost, v.ghost, w.ghost,
            domain.faces.cellid, bc.BCfaces, domain.faces.normal)
      hfunc(u.halo, v.halo, w.halo, u.haloghost, v.haloghost, w.haloghost,
            domain.nodes.haloghostid, domain.ghost.ext_info_int,
            domain.ghost.ext_info_flt, bc.BCtypeindex, domain.halonodes)