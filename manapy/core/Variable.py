#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np
from manapy.domain import Domain
import manapy.backends.types as types
from manapy.boundary.Boundary import Boundary, update_slip_ghost
from types import LambdaType
from manapy.backends.types import FLOAT_TYPE

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
  # Cache of compiled kernels per dimension. Mirrors the old `compile_func`:
  # only the dimension actually used is ever compiled, and only once.
  _compiled_funcs = {}

  @classmethod
  def _get_compiled_funcs(cls, dim, backend):
    key = (backend.name, dim)
    if key in cls._compiled_funcs:
      return cls._compiled_funcs[key]

    # Corps grid-stride unifies (meme source CPU/GPU quand le kernel s'y prete).
    if dim == 2:
      import manapy.core.variable_compute_2d as K
    elif dim == 3:
      import manapy.core.variable_compute_3d as K
    else:
      raise ValueError(f"Unsupported dimension: {dim}")

    funcs = {
      'facetocell': backend.make_gridstride_kernel(K.facetocell, size_arg=1),
      'celltoface': backend.make_gridstride_kernel(K.celltoface, size_arg=(6, 7, 8)),
    }
    if dim == 2:
      funcs['interp']        = backend.make_gridstride_kernel(K.centertovertex_2d, size_arg=13)
      funcs['face_gradient'] = backend.make_gridstride_kernel(K.face_gradient_2d, size_arg=(16, 17, 18, 19, 20))
      funcs['cell_gradient'] = backend.make_gridstride_kernel(K.cell_gradient_2d, size_arg=0)
      funcs['barthlimiter']  = backend.make_gridstride_kernel(K.barthlimiter_2d, size_arg=0)
    elif dim == 3:
      funcs['interp']        = backend.make_gridstride_kernel(K.centertovertex_3d, size_arg=13)
      funcs['face_gradient'] = backend.make_gridstride_kernel(K.face_gradient_3d, size_arg=(16, 17, 18, 19, 20))
      funcs['cell_gradient'] = backend.make_gridstride_kernel(K.cell_gradient_3d, size_arg=0)
      funcs['barthlimiter']  = backend.make_gridstride_kernel(K.barthlimiter_3d, size_arg=0)

    cls._compiled_funcs[key] = funcs
    return funcs

  def __init__(self, domain:Domain, BC:dict=None, values_dict:dict=None, name:str=None):
    if domain is None:
      raise ValueError("domain must be given")

    self._domain = domain
    self.backend = domain.backend
    self._values = values_dict
    self._name = name


    self._dim = domain.dim
    self._nbfaces = domain.nbfaces
    self._nbcells = domain.nbcells
    self._nbnodes = domain.nbnodes
    self._nbhalos = domain.nbhalos

    # Les champs sont alloues DANS la memoire du backend (device sous GPU, host
    # sous CPU) : pas d'allocation host puis transfert.
    _z = self.backend.zeros
    _f = types.np_float_type
    self.cell = _z(self._nbcells, _f)
    self.node = _z(self._nbnodes, _f)
    self.face = _z(self._nbfaces, _f)
    self.ghost = _z(self._nbfaces, _f)  # !! Indexed by face not ghostid
    self.halo = _z(self._nbhalos, _f)

    self.gradcellx = _z(self._nbcells, _f)
    self.gradcelly = _z(self._nbcells, _f)
    self.gradcellz = _z(self._nbcells, _f)

    self.gradhalocellx = _z(self._nbhalos, _f)
    self.gradhalocelly = _z(self._nbhalos, _f)
    self.gradhalocellz = _z(self._nbhalos, _f)

    self.gradfacex = _z(self._nbfaces, _f)
    self.gradfacey = _z(self._nbfaces, _f)
    self.gradfacez = _z(self._nbfaces, _f)

    self.psi = _z(self._nbcells, _f)
    self.psihalo = _z(self._nbhalos, _f)

    self.halotosend = _z(len(domain.halos.halosint), _f)
    self.haloghost = _z(domain.halos.sizehaloghost, _f)

    # TODO these attribute should be declared inside domain class
    # Alloues sur le backend (device sous GPU) : ecrits par les kernels BC.
    self._domain.Pbordnode = _z(self._nbnodes, _f)
    self._domain.Pbordface = _z(self._nbfaces, _f)
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

    # Functions: compile only the kernels needed for this dimension, once.
    funcs = Variable._get_compiled_funcs(self._dim, self.backend)
    self._facetocell    = funcs['facetocell']
    self._celltoface    = funcs['celltoface']
    self._func_interp   = funcs['interp']
    self._face_gradient = funcs['face_gradient']
    self._cell_gradient = funcs['cell_gradient']
    self._barthlimiter  = funcs['barthlimiter']

  def add_term(self, name):
    # Alloue dans la memoire du backend (device sous GPU, host sous CPU).
    self.__dict__[name] = self.backend.zeros(self._nbcells, FLOAT_TYPE)

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
    valueface = np.zeros(self._domain.nbfaces, dtype=types.np_float_type)
    valuenode = np.zeros(self._domain.nbnodes, dtype=types.np_float_type)
    valuehalo = np.zeros(self._domain.halos.sizehaloghost, dtype=types.np_float_type)
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
                              BCvalueface=np.array([],dtype=types.np_float_type),
                              BCvaluenode=np.array([], dtype=types.np_float_type),
                              BCvaluehalo=np.array([], dtype=types.np_float_type),
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
          bc.constNHNode = valuenode

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
          bc.BCvalueface = np.array([], dtype=types.np_float_type)
          bc.BCvaluenode = np.array([], dtype=types.np_float_type)
          bc.BCvaluehalo = np.array([], dtype=types.np_float_type)

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
    neumannfaces = np.asarray(neumannfaces, dtype=types.np_int_type)
    BCneumann = np.asarray(BCneumann, dtype=types.np_int_type)
    dirichletfaces = np.asarray(dirichletfaces, dtype=types.np_int_type)
    BCdirichlet = np.asarray(BCdirichlet, dtype=types.np_int_type)
    neumannNHfaces = np.asarray(neumannNHfaces, dtype=types.np_int_type)
    BCneumannNH = np.asarray(BCneumannNH, dtype=types.np_int_type)

    if self.backend.name == "gpu":
      neumannfaces = self.backend.asarray(neumannfaces, types.np_int_type)
      BCneumann = self.backend.asarray(BCneumann, types.np_int_type)
      dirichletfaces = self.backend.asarray(dirichletfaces, types.np_int_type)
      BCdirichlet = self.backend.asarray(BCdirichlet, types.np_int_type)
      neumannNHfaces = self.backend.asarray(neumannNHfaces, types.np_int_type)
      BCneumannNH = self.backend.asarray(BCneumannNH, types.np_int_type)

      converted_bc_arrays = {}
      for bc in BCs.values():
        if bc is None:
          continue
        for attr in ("BCvalueface", "BCvaluenode", "BCvaluehalo",
                     "constNH", "constNHNode"):
          value = getattr(bc, attr, None)
          if isinstance(value, np.ndarray):
            key = id(value)
            device_value = converted_bc_arrays.get(key)
            if device_value is None:
              device_value = self.backend.asarray(value, value.dtype)
              converted_bc_arrays[key] = device_value
            setattr(bc, attr, device_value)

    return (neumannfaces,
            BCneumann,
            dirichletfaces,
            BCdirichlet,
            neumannNHfaces,
            BCneumannNH,
            BCs)

  def __str__(self):
    """Pretty printing"""
    txt = '\n'
    txt += '> dim   :: {dim}\n'.format(dim=self._dim)
    txt += '> total cells  :: {cells}\n'.format(cells=self._nbcells)
    txt += '> total nodes  :: {nodes}\n'.format(nodes=self._nbnodes)
    txt += '> total faces  :: {faces}\n'.format(faces=self._nbfaces)
    txt += '> Value on cells :: {faces}\n'.format(faces=self.cell)
    txt += '> Value on faces  :: {faces}\n'.format(faces=self.face)
    txt += '> Value on nodes  :: {faces}\n'.format(faces=self.node)
    txt += '> Value on ghosts  :: {faces}\n'.format(faces=self.ghost)

    return txt

  def __add__(self, other):

    # TODO should this has its own BC, name, values_dict
    res = Variable(self._domain)

    res.cell = self.cell + other.cell

    return res

  def __sub__(self, other):

    res = Variable(self._domain)

    res.cell = self.cell - other.cell

    return res

  def __mul__(self, other):

    res = Variable(self._domain)

    res.cell = self.cell * other.cell

    return res

  def __truediv__(self, other):

    res = Variable(self._domain)

    if np.all(other.cell != 0):
      res.cell = self.cell / other.cell
    else:
      raise ValueError("Values of denominator must be different to 0")


    return res

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
                         self._domain.ghost.ext_info_int, self._domain.ghost.ext_info_flt, BC.BCtypeindex, self._domain.halonodes, BC.constNHNode)

    # Coupled slip walls: apply the reflection on the whole velocity group that
    # auto-registered on the domain (in creation order u, v[, w]).
    if self._has_slip:
      update_slip_ghost(self._domain.slip_velocity)

  def norml2(self, exact, order=None):

    if order is None:
      order = 1
    assert self._nbcells == len(exact), 'exact solution must have length of cells'

    Error = np.zeros(self._nbcells, dtype=types.np_float_type)
    Ex = np.zeros(self._nbcells, dtype=types.np_float_type)

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