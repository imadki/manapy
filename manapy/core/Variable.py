#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np

from manapy.backends import ManapyArray
from manapy.backends.ManapyArray import Device
from manapy.backends.config import ManapyConfig
from manapy.domain import Domain
from manapy.boundary.Boundary import Boundary, update_slip_ghost
from types import LambdaType
from manapy.compute import VariableCompute

"""
# self, domain=None, terms=None, comm=None, name=None, BC=None, values=None, *args, **kwargs

# - terms is a list of names that will be added to the Variable Class as self.__dict__[term] = np.array(nb_cells)
# - comm not used
# - name of the variable
# - BC is a dic that always have in 2D (in, out, upper, bottom) 3D (..., front, back)
  # values: dirichlet, neumann, noslip, neumannNH
# - values is a dic string => (lambda or int) keys: [in, out, upper, bottom, front, back]
# - *args, **kwargs not used

# terms is replaced with a method
# removed
    self.backend = self._domain.backend
    self.signature = self._domain.signature
    self.float_precision = self._domain.float_precision
    self.int_precision = self._domain.int_precision
    self.mpi_precision = self._domain.mpi_precision
    self.forcedbackend = self._domain.forcedbackend

is_called = False
"""

class Variable:
  # Available slope limiters for the 2nd-order (MUSCL) gradient reconstruction.
  # 'barth'  : Barth-Jespersen min(1, y) (default; == minmod in this multi-D
  #            neighbourhood-min/max framework, where the argument y is always >=0).
  # 'vanalbada'/'venkatakrishnan': smooth phi(y)=(y^2+2y)/(y^2+y+2) -- less clipping
  #            of smooth extrema, smoother convergence. Same kernel signature.
  _LIMITER_KERNELS = {'barth': 'barthlimiter', 'vanalbada': 'vanalbadalimiter',
                      'venkatakrishnan': 'vanalbadalimiter'}


  def __init__(self, domain:Domain, BC:dict=None, values_dict:dict=None, name:str=None,
               limiter:str='barth'):
    if domain is None:
      raise ValueError("domain must be given")

    self._limiter = str(limiter).lower() if limiter else 'barth'
    if self._limiter == 'none':
      self._limiter = 'barth'
    self._domain = domain
    self.config = self._domain.config
    self._values = values_dict
    self._name = name
    self.compute = VariableCompute(self.config, self.domain.dim)


    self._dim = domain.dim
    self._nbfaces = domain.nbfaces
    self._nbcells = domain.nbcells
    self._nbnodes = domain.nbnodes
    self._nbhalos = domain.nbhalos

    float_precision = self.config.float_dtype
    device = self.config.device
    self.cell = ManapyArray.zeros(self._nbcells, float_precision, device)
    self.node = ManapyArray.zeros(self._nbnodes, float_precision, device)
    self.face = ManapyArray.zeros(self._nbfaces, float_precision, device)
    self.ghost = ManapyArray.zeros(self._nbfaces, float_precision, device)  # !! Indexed by face not ghostid
    self.halo = ManapyArray.zeros(self._nbhalos, float_precision, device)

    self.gradcellx = ManapyArray.zeros(self._nbcells, float_precision, device)
    self.gradcelly = ManapyArray.zeros(self._nbcells, float_precision, device)
    self.gradcellz = ManapyArray.zeros(self._nbcells, float_precision, device)

    self.gradhalocellx = ManapyArray.zeros(self._nbhalos, float_precision, device)
    self.gradhalocelly = ManapyArray.zeros(self._nbhalos, float_precision, device)
    self.gradhalocellz = ManapyArray.zeros(self._nbhalos, float_precision, device)

    self.gradfacex = ManapyArray.zeros(self._nbfaces, float_precision, device)
    self.gradfacey = ManapyArray.zeros(self._nbfaces, float_precision, device)
    self.gradfacez = ManapyArray.zeros(self._nbfaces, float_precision, device)

    self.psi = ManapyArray.zeros(self._nbcells, float_precision, device)
    self.psihalo = ManapyArray.zeros(self._nbhalos, float_precision, device)

    self.halotosend = ManapyArray.zeros(len(domain.halos.halosint), float_precision, device)
    self.haloghost = ManapyArray.zeros(domain.halos.sizehaloghost, float_precision, device)

    # Alloues sur le backend (device sous GPU) : ecrits par les kernels BC.
    # these are ManapyArray
    (self.neumannfaces,
    self.BCneumann,
    self.dirichletfaces,
    self.BCdirichlet,
    self.neumannNHfaces,
    self.BCneumannNH,
    self._BCs) = self._update_boundaries(BC, self._values)

    # A "slip" wall is a coupled (vector) BC: it needs every velocity component
    # together. Variables carrying a slip BC auto-register on the domain in
    # creation order (u, v[, w]), so update_ghost_value() can apply the coupled
    # slip automatically without the user wiring update_slip_ghost() by hand.
    self._has_slip = any(bc is not None and bc.BCtype in Boundary._VECTOR_TYPES
                         for bc in self._BCs.values())
    if self._has_slip:
      if getattr(self._domain, "slip_velocity", None) is None:
        self._domain.slip_velocity = []
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
          print("WARNING: 'slip' boundary detected (coupled vector BC).\n"
                "         Create the velocity components in spatial order "
                "(u, v[, w]), each with the slip BC;\n"
                "         only velocity components must carry 'slip'. They "
                "auto-register and the reflection\n"
                "         is applied by update_ghost_value() in that order.")
      self._domain.slip_velocity.append(self)

    self._BCin = self.BCs["in"]
    self._BCout = self.BCs["out"]
    self._BCbottom = self.BCs["bottom"]
    self._BCupper = self.BCs["upper"]
    self._BCfront = None
    self._BCback = None
    if self.dim == 3:
      self._BCfront = self.BCs["front"]
      self._BCback = self.BCs["back"]


    self._facetocell    = self.compute.facetocell
    self._celltoface    = self.compute.celltoface
    self._func_interp   = self.compute.interp
    self._face_gradient = self.compute.face_gradient
    self._cell_gradient = self.compute.cell_gradient
    self._barthlimiter = self.compute.barthlimiter
    if self._limiter == 'vanalbada' or self._limiter == 'venkatakrishnan':
      self._barthlimiter = self.compute.vanalbadalimiter

  def add_term(self, name):
    # Alloue dans la memoire du backend (device sous GPU, host sous CPU).
    self.__dict__[name] = ManapyArray.zeros(self._nbcells, self.config.float_dtype, self.config.device)

  def _fill_bc_values(self, value, bcfaces, bctypeindex, valueface, valuenode, valuehalo):
    """Fill the face / node / haloghost boundary arrays from a prescribed value.

    `value` may be a constant (int/float) or a callable ``lambda x, y, z``.
    Shared by the dirichlet and neumannNH branches of `_update_boundaries`.
    """
    bcnodes = np.where(self._domain.nodes.oldname == bctypeindex)[0]

    # Flatten the ragged (node -> haloghost ids) map for the boundary nodes:
    # the last column holds the per-node count, the others the ghost ids.
    haloghostid = self._domain.nodes.haloghostid
    if len(bcnodes):
      maxg = haloghostid.shape[1] - 1
      counts = haloghostid[bcnodes, -1]
      mask = np.arange(maxg)[None, :] < counts[:, None]
      ghost_ids = haloghostid[bcnodes][:, :maxg][mask]
    else:
      ghost_ids = np.empty(0, dtype=haloghostid.dtype)

    fc = self._domain.faces.center
    vx = self._domain.nodes.vertex
    gfc = self._domain.ghost.ext_info_flt

    if isinstance(value, LambdaType):
      try:
        valueface[bcfaces] = value(fc[bcfaces, 0], fc[bcfaces, 1], fc[bcfaces, 2])
        valuenode[bcnodes] = value(vx[bcnodes, 0], vx[bcnodes, 1], vx[bcnodes, 2])
        if len(ghost_ids):
          valuehalo[ghost_ids] = value(gfc[ghost_ids, 4], gfc[ghost_ids, 5], gfc[ghost_ids, 6])
      except (TypeError, ValueError):
        # lambda not vectorizable over arrays: evaluate element by element
        for i in bcfaces:
          valueface[i] = value(fc[i, 0], fc[i, 1], fc[i, 2])
        for i in bcnodes:
          valuenode[i] = value(vx[i, 0], vx[i, 1], vx[i, 2])
        for gid in ghost_ids:
          valuehalo[gid] = value(gfc[gid, 4], gfc[gid, 5], gfc[gid, 6])

    elif isinstance(value, (int, float)):
      valueface[bcfaces] = value
      valuenode[bcnodes] = value
      if len(ghost_ids):
        valuehalo[ghost_ids] = value

    else:
      raise ValueError("BC value must be a number or a callable lambda(x, y, z)")

  def _update_boundaries(self, BC:dict, values_dict:dict):
    valueface = np.zeros(self._domain.nbfaces, dtype=self.config.float_dtype)
    valuenode = np.zeros(self._domain.nbnodes, dtype=self.config.float_dtype)
    valuehalo = np.zeros(self._domain.halos.sizehaloghost, dtype=self.config.float_dtype)
    # Constantes de BC : remplies sur host (_fill_bc_values) ; transferees au bord
    # des kernels (read-only). Pas de wrap GPUArray.

    neumannfaces = []
    BCneumann = []
    dirichletfaces = []
    BCdirichlet = []
    neumannNHfaces = []
    BCneumannNH = []

    BCs : dict[str, Boundary] = {"in":  None, "out": None, "bottom": None, "upper": None}
    if self._dim == 3:
      BCs = {"in": None, "out": None, "bottom": None, "upper": None, "front": None, "back": None}
    domain_BCs = self._domain.BCs # See LocalDomainClass._define_BCs

    if BC is None:
      for loc in BCs.keys():
        domain_bc_typename = domain_BCs[loc][0]
        domain_bc_type_idx = domain_BCs[loc][1]
        if domain_bc_typename == "periodic":
          BCs[loc] = Boundary(BCtype="periodic",
                              BCloc=loc,
                              BCvalueface=np.array([],dtype=self.config.float_dtype),
                              BCvaluenode=np.array([], dtype=self.config.float_dtype),
                              BCvaluehalo=np.array([], dtype=self.config.float_dtype),
                              BCtypeindex=domain_bc_type_idx,
                              domain=self._domain)

        elif domain_bc_typename == "neumann":
          BCs[loc] = Boundary(BCtype="neumann",
                              BCloc=loc,
                              BCvalueface=self.cell,
                              BCvaluenode=self.cell,
                              BCvaluehalo=self.halo,
                              BCtypeindex=domain_bc_type_idx,
                              domain=self._domain)

          BCneumann.append(BCs[loc].BCtypeindex)
          neumannfaces.extend(BCs[loc].BCfaces)

          valueface = self.cell
          valuenode = self.node
          valuehalo = self.halo
        else:
          raise RuntimeError("Unknown BCtype")

        BCs[loc].BCvalueface = valueface
        BCs[loc].BCvaluenode = valuenode
        BCs[loc].BCvaluehalo = valuehalo

    else:
      for loc, bct in BC.items():
        domain_bc_typename = domain_BCs[loc][0]
        domain_bc_type_idx = domain_BCs[loc][1]
        if domain_bc_typename == "periodic":
          if bct != "periodic":
            raise ValueError("BC must be periodic for " + str(loc))

        elif domain_bc_typename != "periodic":
          if bct == "periodic":
            raise ValueError("BC must be not periodic for " + str(loc))

        # Build the Boundary once: the constructor arguments are identical for
        # every BC type (only BCtype differs).
        BCs[loc] = Boundary(BCtype=bct,
                            BCloc=loc,
                            BCvalueface=self.cell,
                            BCvaluenode=self.cell,
                            BCvaluehalo=self.halo,
                            BCtypeindex=domain_bc_type_idx,
                            domain=self._domain)
        bc = BCs[loc]

        if bct == "dirichlet":
          BCdirichlet.append(bc.BCtypeindex)
          dirichletfaces.extend(bc.BCfaces)

          if values_dict is None or loc not in values_dict.keys():
            raise ValueError("Value of dirichlet BC for " + str(loc) + " faces must be given")

          # TODO check valuehalo (face center miss)
          self._fill_bc_values(values_dict[loc], bc.BCfaces, bc.BCtypeindex,
                               valueface, valuenode, valuehalo)

          bc.BCvalueface = valueface
          bc.BCvaluenode = valuenode
          bc.BCvaluehalo = valuehalo

        elif bct == "neumannNH":
          BCneumannNH.append(bc.BCtypeindex)
          neumannNHfaces.extend(bc.BCfaces)

          if values_dict is None or loc not in values_dict.keys():
            raise ValueError("Value of neumannNH BC for " + str(loc) + " faces must be given")

          # TODO check valuehalo (face center miss)
          self._fill_bc_values(values_dict[loc], bc.BCfaces, bc.BCtypeindex,
                               valueface, valuenode, valuehalo)

          bc.constNH = valueface
          # Per halo ghost, evaluated at the ghost's face centre -- the halo
          # counterpart of valueface. A per-node array cannot be used here: a
          # halo ghost is reachable from every node of its face (and corner
          # nodes carry a neighbouring boundary's tag), so the gradient applied
          # would depend on which node reached the ghost last.
          bc.constNHGhost = valuehalo

          bc.BCvalueface = self.cell
          bc.BCvaluenode = self.node
          bc.BCvaluehalo = self.halo

        elif bct == "neumann":
          BCneumann.append(bc.BCtypeindex)
          neumannfaces.extend(bc.BCfaces)

          bc.BCvalueface = self.cell  # TODO why not self.face
          bc.BCvaluenode = self.node
          bc.BCvaluehalo = self.halo

        elif bct == "periodic":
          bc.BCvalueface = np.array([], dtype=self.config.float_dtype)
          bc.BCvaluenode = np.array([], dtype=self.config.float_dtype)
          bc.BCvaluehalo = np.array([], dtype=self.config.float_dtype)

        elif bct == "slip":
          BCneumann.append(bc.BCtypeindex)
          neumannfaces.extend(bc.BCfaces)

          bc.BCvalueface = self.cell
          bc.BCvaluenode = self.node
          bc.BCvaluehalo = self.halo

        elif bct == "nonslip":
          BCneumann.append(bc.BCtypeindex)
          neumannfaces.extend(bc.BCfaces)

          bc.BCvalueface = self.cell
          bc.BCvaluenode = self.node
          bc.BCvaluehalo = self.halo

        else:
          raise ValueError(f"Invalid BCtype {bct}")


    neumannfaces.sort()
    dirichletfaces.sort()
    neumannfaces = ManapyArray(np.asarray(neumannfaces, dtype=self.config.int_dtype), self.config.device)
    BCneumann = ManapyArray(np.asarray(BCneumann, dtype=self.config.int_dtype), self.config.device)
    dirichletfaces = ManapyArray(np.asarray(dirichletfaces, dtype=self.config.int_dtype), self.config.device)
    BCdirichlet = ManapyArray(np.asarray(BCdirichlet, dtype=self.config.int_dtype), self.config.device)
    neumannNHfaces = ManapyArray(np.asarray(neumannNHfaces, dtype=self.config.int_dtype), self.config.device)
    BCneumannNH = ManapyArray(np.asarray(BCneumannNH, dtype=self.config.int_dtype), self.config.device)


    return (neumannfaces,
            BCneumann,
            dirichletfaces,
            BCdirichlet,
            neumannNHfaces,
            BCneumannNH,
            BCs)

  def update_halo_value(self):
    # update the halo values
    self.domain.halo_comm.exchange(self.cell, recv_buffer=self.halo)


  def interpolate_facetocell(self):
    self._facetocell(self.face, self.cell, self._domain.cells.faceid, self._dim)

  def interpolate_celltoface(self):
    self._celltoface(self.cell, self.face, self.ghost, self.halo, self._domain.faces.cellid,
                         self._domain.faces.halofid,
                         self._domain.innerfaces, self._domain.boundaryfaces, self._domain.halofaces)

  def interpolate_celltonode(self):
    # self.update_halo_value()
    # self.update_ghost_value()
    self._func_interp(self.cell, self.ghost, self.halo, self.haloghost, self._domain.cells.center,
                          self._domain.halos.centvol,
                          self._domain.nodes.cellid, self._domain.ghost.info_flt, self._domain.ghost.ext_info_flt, self._domain.nodes.ghostid,
                          self._domain.nodes.haloghostid,
                          self._domain.nodes.periodicid, self._domain.nodes.halonid, self._domain.nodes.vertex,
                          self._domain.nodes.oldname,
                          self._domain.nodes.R_x, self._domain.nodes.R_y, self._domain.nodes.R_z,
                          self._domain.nodes.lambda_x,
                          self._domain.nodes.lambda_y, self._domain.nodes.lambda_z,
                          self._domain.nodes.number, self._domain.cells.shift, self.node, self.domain.ghost.faceid)

  def compute_cell_gradient(self):
    self._cell_gradient(self.cell, self.ghost, self.halo, self.haloghost, self._domain.cells.center,
                            self._domain.cells.cellnid, self._domain.ghost.info_flt, self._domain.ghost.ext_info_flt, self._domain.cells.ghostnid, self._domain.cells.haloghostnid,
                            self._domain.cells.halonid, self._domain.cells.nodeid, self._domain.cells.periodicnid,
                            self._domain.nodes.periodicid,
                            self._domain.nodes.oldname, self._domain.halos.centvol, self._domain.cells.shift,
                            self.gradcellx,
                            self.gradcelly, self.gradcellz, self.domain.ghost.faceid)

    # The limiter depend on hc value
    self._barthlimiter(self.cell, self.ghost, self.halo, self.gradcellx, self.gradcelly, self.gradcellz,
                           self.psi, self._domain.faces.cellid, self._domain.cells.faceid, self._domain.faces.name,
                           self._domain.faces.halofid, self._domain.cells.center, self._domain.faces.center)

    self.domain.halo_comm.graph_comm.Barrier()
    # update the halo values
    self.domain.halo_comm.exchange(self.gradcellx, recv_buffer=self.gradhalocellx)
    self.domain.halo_comm.exchange(self.gradcelly, recv_buffer=self.gradhalocelly)
    self.domain.halo_comm.exchange(self.gradcellz, recv_buffer=self.gradhalocellz)
    self.domain.halo_comm.exchange(self.psi, recv_buffer=self.psihalo)

  def compute_face_gradient(self):

    self._face_gradient(self.cell, self.ghost, self.halo, self.node, self._domain.faces.cellid,
                            self._domain.faces.nodeid,
                            self._domain.faces.halofid, self._domain.faces.airDiamond,
                            self._domain.faces.normal, self._domain.faces.f_1, self._domain.faces.f_2,
                            self._domain.faces.f_3, self._domain.faces.f_4,
                            self.gradfacex, self.gradfacey, self.gradfacez, self._domain.innerfaces,
                            self._domain.halofaces, self.dirichletfaces, self.neumannfaces,
                            self._domain.periodicboundaryfaces)

  def update_ghost_value(self):
    for BC in self._BCs.values():
      # vector (component-coupled) BCs such as slip have no per-scalar kernel;
      # they are applied below via the coupled update_slip_ghost.
      if BC.func_ghost is None:
        continue
      BC.func_ghost(BC.BCvalueface, self.ghost, self._domain.faces.cellid,
                     BC.BCfaces,
                     BC.constNH, self._domain.faces.dist_ortho)
      BC.func_haloghost(BC.BCvaluehalo, self.haloghost, self._domain.nodes.haloghostid,
                         self._domain.ghost.ext_info_int, self._domain.ghost.ext_info_flt, BC.BCtypeindex, self._domain.halonodes, BC.constNHGhost)

    # Coupled slip walls: apply the reflection on the whole velocity group that
    # auto-registered on the domain (in creation order u, v[, w]).
    if self._has_slip:
      update_slip_ghost(self._domain.slip_velocity)

  def norml2(self, exact, order=None):

    if order is None:
      order = 1
    assert self._nbcells == len(exact), 'exact solution must have length of cells'

    Error = np.zeros(self._nbcells, dtype=self.config.float_dtype)
    Ex = np.zeros(self._nbcells, dtype=self.config.float_dtype)

    for i in range(len(exact)):
      Error[i] = np.fabs(self.cell[i] - exact[i]) * self._domain.cells.volume[i]
      Ex[i] = np.fabs(exact[i]) * self._domain.cells.volume[i]

    ErrorL2 = np.linalg.norm(Error, ord=order) / np.linalg.norm(Ex, ord=order)

    return ErrorL2

  @property
  def domain(self) -> Domain:
    return self._domain

  @property
  def dim(self):
    return self._dim


  @property
  def nbfaces(self):
    return self._nbfaces

  @property
  def nbcells(self):
    return self._nbcells

  @property
  def nbnodes(self):
    return self._nbnodes

  @property
  def nbhalos(self):
    return self._nbhalos

  @property
  def name(self):
    return self._name

  @property
  def BCs(self) -> dict[str, Boundary]:
    return self._BCs

  @property
  def BCin(self):
    return self._BCin

  @property
  def BCout(self):
    return self._BCout

  @property
  def BCupper(self):
    return self._BCupper

  @property
  def BCbottom(self):
    return self._BCbottom

  @property
  def BCback(self):
    return self._BCback

  @property
  def BCfront(self):
    return self._BCfront