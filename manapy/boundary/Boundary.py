from manapy.domain import Domain
import manapy.boundary.bc_compute as bc_compute
import numpy as np
import manapy.backends.types as types
from manapy.backends.compile_fun import compile

class Boundary:
  # Cache of compiled (ghost, haloghost) kernels per BC kind. Mirrors the old
  # compile_func: only the boundary types actually used are ever compiled.
  _compiled = {}

  @classmethod
  def _get_funcs(cls, kind):
    if kind in cls._compiled:
      return cls._compiled[kind]
    mapping = {
      'neumann':   (bc_compute._ghost_value_neumann,   bc_compute._haloghost_value_neumann),
      'dirichlet': (bc_compute._ghost_value_dirichlet, bc_compute._haloghost_value_dirichlet),
      'neumannNH': (bc_compute._ghost_value_neumannNH, bc_compute._haloghost_value_neumannNH),
      'nonslip':   (bc_compute._ghost_value_nonslip,   bc_compute._haloghost_value_nonslip),
    }
    g, h = mapping[kind]
    cls._compiled[kind] = (compile(g), compile(h))
    return cls._compiled[kind]

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

    self._func_ghost_args = []
    self._func_haloghost_args = []

    self.constNH = np.zeros(1, dtype=types.np_float_type)
    self.constNHNode = np.zeros(1, dtype=types.np_float_type)

    if BCloc == "in":
      self._BCfaces = self._domain.infaces
      self._BCnodes = self._domain.innodes
      self._BCtypeindex = BCtypeindex
    elif BCloc == "out":
      self._BCfaces = self._domain.outfaces
      self._BCnodes = self._domain.outnodes
      self._BCtypeindex = BCtypeindex
    elif BCloc == "bottom":
      self._BCfaces = self._domain.bottomfaces
      self._BCnodes = self._domain.bottomnodes
      self._BCtypeindex = BCtypeindex
    elif BCloc == "upper":
      self._BCfaces = self._domain.upperfaces
      self._BCnodes = self._domain.uppernodes
      self._BCtypeindex = BCtypeindex
    elif BCloc == "front":
      self._BCfaces = self._domain.frontfaces
      self._BCnodes = self._domain.frontnodes
      self._BCtypeindex = BCtypeindex
    elif BCloc == "back":
      self._BCfaces = self._domain.backfaces
      self._BCnodes = self._domain.backnodes
      self._BCtypeindex = BCtypeindex
    else:
      raise ValueError(f"unknown BCloc: {BCloc}")

    if self._BCtype == "neumann" or self._BCtype == "periodic":
      self._func_ghost, self._func_haloghost = Boundary._get_funcs('neumann')
    elif self._BCtype == "dirichlet":
      self._func_ghost, self._func_haloghost = Boundary._get_funcs('dirichlet')
    elif self._BCtype == "neumannNH":
      self._func_ghost, self._func_haloghost = Boundary._get_funcs('neumannNH')
    elif self._BCtype == "nonslip":
      self._func_ghost, self._func_haloghost = Boundary._get_funcs('nonslip')
    else:
      raise ValueError(f"unknown BCtype: {BCtype}")
    # elif self._BCtype == "slip":
    #   self._func_ghost = ghost_value_slip
    #   self._func_haloghost = haloghost_value_slip
    #   self._func_ghost_args.extend([self._BCvaluefacetmp, self.domain.faces.normal, self.domain.faces.mesure])
    #   self._func_haloghost_args.extend([self._BCvaluehalotmp, self.domain.nodes.ghostfaceinfo])

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