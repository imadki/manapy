#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np
from manapy.domain import Domain
import manapy.backends.types as types
from manapy.boundary.Boundary import Boundary
from types import LambdaType
import manapy.core.variable_compute_2d as variable_compute_2d
import manapy.core.variable_compute_3d as variable_compute_3d
import manapy.core.variable_compute_utils as variable_compute_utils
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
  def __init__(self, domain:Domain, BC:dict=None, values_dict:dict=None, name:str=None):
    if domain is None:
      raise ValueError("domain must be given")

    self._domain = domain
    self._values = values_dict
    self._name = name


    self._dim = domain.dim
    self._nbfaces = domain.nbfaces
    self._nbcells = domain.nbcells
    self._nbnodes = domain.nbnodes
    self._nbhalos = domain.nbhalos
    self._nbghost = domain.nbghost

    self.cell = np.zeros(self._nbcells, dtype=types.np_float_type)
    self.node = np.zeros(self._nbnodes, dtype=types.np_float_type)
    self.face = np.zeros(self._nbfaces, dtype=types.np_float_type)
    self.ghost = np.zeros(self._nbghost, dtype=types.np_float_type)
    self.halo = np.zeros(self._nbhalos, dtype=types.np_float_type)

    self.gradcellx = np.zeros(self._nbcells, dtype=types.np_float_type)
    self.gradcelly = np.zeros(self._nbcells, dtype=types.np_float_type)
    self.gradcellz = np.zeros(self._nbcells, dtype=types.np_float_type)

    self.gradhalocellx = np.zeros(self._nbhalos, dtype=types.np_float_type)
    self.gradhalocelly = np.zeros(self._nbhalos, dtype=types.np_float_type)
    self.gradhalocellz = np.zeros(self._nbhalos, dtype=types.np_float_type)

    self.gradfacex = np.zeros(self._nbfaces, dtype=types.np_float_type)
    self.gradfacey = np.zeros(self._nbfaces, dtype=types.np_float_type)
    self.gradfacez = np.zeros(self._nbfaces, dtype=types.np_float_type)

    self.psi = np.zeros(self._nbcells, dtype=types.np_float_type)
    self.psihalo = np.zeros(self._nbhalos, dtype=types.np_float_type)

    self.halotosend = np.zeros(len(domain.halos.halosint), dtype=types.np_float_type)
    self.haloghost = np.zeros(domain.halos.sizehaloghost, dtype=types.np_float_type)

    # TODO these attribute should be declared inside domain class
    self._domain.Pbordnode = np.zeros(self._domain.nbnodes, dtype=types.np_float_type)
    self._domain.Pbordface = np.zeros(self._domain.nbfaces, dtype=types.np_float_type)
    (self.neumannfaces,
    self.BCneumann,
    self.dirichletfaces,
    self.BCdirichlet,
    self.neumannNHfaces,
    self.BCneumannNH,
    self._BCs) = self._update_boundaries(BC, self._values)

    self._BCin = self.BCs["in"]
    self._BCout = self.BCs["out"]
    self._BCbottom = self.BCs["bottom"]
    self._BCupper = self.BCs["upper"]
    self._BCfront = None
    self._BCback = None
    if self.dim == 3:
      self._BCfront = self.BCs["front"]
      self._BCback = self.BCs["back"]

    # Functions
    self._facetocell = variable_compute_utils.facetocell
    self._celltoface = variable_compute_utils.celltoface
    if self._dim == 2:
      self._func_interp = variable_compute_2d.centertovertex_2d
      self._face_gradient = variable_compute_2d.face_gradient_2d
      self._cell_gradient = variable_compute_2d.cell_gradient_2d
      self._barthlimiter = variable_compute_2d.barthlimiter_2d
    elif self._dim == 3:
      self._func_interp = variable_compute_3d.centertovertex_3d
      self._face_gradient = variable_compute_3d.face_gradient_3d
      self._cell_gradient = variable_compute_3d.cell_gradient_3d
      self._barthlimiter = variable_compute_3d.barthlimiter_3d

  def add_term(self, name):
    self.__dict__[name] = np.zeros(self._nbcells, dtype=FLOAT_TYPE)

  def _update_boundaries(self, BC:dict, values_dict:dict):
    valueface = np.zeros(self._domain.nbfaces, dtype=types.np_float_type)
    valuenode = np.zeros(self._domain.nbnodes, dtype=types.np_float_type)
    valuehalo = np.zeros(self._domain.halos.sizehaloghost, dtype=types.np_float_type)

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

        if bct == "dirichlet":  # or bct =="neumannNH":
          BCs[loc] = Boundary(BCtype=bct,
                              BCloc=loc,
                              BCvalueface=self.cell,
                              BCvaluenode=self.cell,
                              BCvaluehalo=self.halo,
                              BCtypeindex=domain_bc_type_idx,
                              domain=self._domain)

          BCdirichlet.append(BCs[loc].BCtypeindex)
          dirichletfaces.extend(BCs[loc].BCfaces)

          if values_dict is None or loc not in values_dict.keys():
            raise ValueError("Value of dirichlet BC for " + str(loc) + " faces must be given")

          # TODO check valuehalo (face center miss)
          if isinstance(values_dict[loc], LambdaType):
            for i in BCs[loc].BCfaces:
              valueface[i] = values_dict[loc](self._domain.faces.center[i][0], self._domain.faces.center[i][1],
                                               self._domain.faces.center[i][2])
            for i in np.where(self._domain.nodes.oldname == BCs[loc].BCtypeindex)[0]:
              valuenode[i] = values_dict[loc](self._domain.nodes.vertex[i][0], self._domain.nodes.vertex[i][1],
                                               self._domain.nodes.vertex[i][2])

             
              for j in range(self._domain.nodes.haloghostid[i, -1]):
                ghost_id = self._domain.nodes.haloghostid[i, j]
                face_center = self._domain.ghost.ext_info_flt[ghost_id][4:7]
                valuehalo[ghost_id] = values_dict[loc](face_center[0], face_center[1], face_center[2])

          elif isinstance(values_dict[loc], (int, float)):
            for i in BCs[loc].BCfaces:
              valueface[i] = values_dict[loc]

            for i in np.where(self._domain.nodes.oldname == BCs[loc].BCtypeindex)[0]:
              valuenode[i] = values_dict[loc]

              for j in range(self._domain.nodes.haloghostid[i, -1]):
                ghost_id = self._domain.nodes.haloghostid[i, j]
                valuehalo[ghost_id] = values_dict[loc]

          BCs[loc].BCvalueface = valueface
          BCs[loc].BCvaluenode = valuenode
          BCs[loc].BCvaluehalo = valuehalo

        elif bct == "neumannNH":
          BCs[loc] = Boundary(BCtype=bct,
                              BCloc=loc,
                              BCvalueface=self.cell,
                              BCvaluenode=self.cell,
                              BCvaluehalo=self.halo,
                              BCtypeindex=domain_bc_type_idx,
                              domain=self._domain)

          BCneumannNH.append(BCs[loc].BCtypeindex)
          neumannNHfaces.extend(BCs[loc].BCfaces)

          if values_dict is None or loc not in values_dict.keys():
            raise ValueError("Value of dirichlet BC for " + str(loc) + " faces must be given")

          # TODO check valuehalo (face center miss)
          if isinstance(values_dict[loc], LambdaType):
            for i in BCs[loc].BCfaces:
              valueface[i] = values_dict[loc](self._domain.faces.center[i][0], self._domain.faces.center[i][1],
                                               self._domain.faces.center[i][2])
            for i in np.where(self._domain.nodes.oldname == BCs[loc].BCtypeindex)[0]:
              valuenode[i] = values_dict[loc](self._domain.nodes.vertex[i][0], self._domain.nodes.vertex[i][1],
                                               self._domain.nodes.vertex[i][2])

              for j in range(self._domain.nodes.haloghostid[i, -1]):
                ghost_id = self._domain.nodes.haloghostid[i, j]
                face_center = self._domain.ghost.ext_info_flt[ghost_id][4:7]
                valuehalo[ghost_id] = values_dict[loc](face_center[0], face_center[1], face_center[2])

          elif isinstance(values_dict[loc], (int, float)):
            for i in BCs[loc].BCfaces:
              valueface[i] = values_dict[loc]

            for i in np.where(self._domain.nodes.oldname == BCs[loc].BCtypeindex)[0]:
              valuenode[i] = values_dict[loc]

              for j in range(self._domain.nodes.haloghostid[i, -1]):
                ghost_id = self._domain.nodes.haloghostid[i, j]
                valuehalo[ghost_id] = values_dict[loc]

          BCs[loc].constNH = valueface
          BCs[loc].constNHNode = valuenode

          valueface2 = self.cell
          valuenode2 = self.node
          valuehalo2 = self.halo

          BCs[loc].BCvalueface = valueface2
          BCs[loc].BCvaluenode = valuenode2
          BCs[loc].BCvaluehalo = valuehalo2


        elif bct == "neumann":
          BCs[loc] = Boundary(BCtype=bct,
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

          BCs[loc].BCvalueface = valueface
          BCs[loc].BCvaluenode = valuenode
          BCs[loc].BCvaluehalo = valuehalo

        elif bct == "periodic":
          BCs[loc] = Boundary(BCtype=bct,
                              BCloc=loc,
                              BCvalueface=self.cell,
                              BCvaluenode=self.cell,
                              BCvaluehalo=self.halo,
                              BCtypeindex=domain_bc_type_idx,
                              domain=self._domain)

          BCs[loc].BCvalueface = np.array([], dtype=types.np_float_type)
          BCs[loc].BCvaluenode = np.array([], dtype=types.np_float_type)
          BCs[loc].BCvaluehalo = np.array([], dtype=types.np_float_type)


        elif bct == "slip":
          BCs[loc] = Boundary(BCtype=bct,
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

          BCs[loc].BCvalueface = valueface
          BCs[loc].BCvaluenode = valuenode
          BCs[loc].BCvaluehalo = valuehalo


        elif bct == "nonslip":
          BCs[loc] = Boundary(BCtype=bct,
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

          BCs[loc].BCvalueface = valueface
          BCs[loc].BCvaluenode = valuenode
          BCs[loc].BCvaluehalo = valuehalo

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
                         self._domain.innerfaces, self._domain.boundaryfaces, self._domain.halofaces, self._domain.faces.ghost_id)

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
                          self._domain.nodes.number, self._domain.cells.shift, self.node)

  def compute_cell_gradient(self):
    self._cell_gradient(self.cell, self.ghost, self.halo, self.haloghost, self._domain.cells.center,
                            self._domain.cells.cellnid, self._domain.ghost.info_flt, self._domain.ghost.ext_info_flt, self._domain.cells.ghostnid, self._domain.cells.haloghostnid,
                            self._domain.cells.halonid, self._domain.cells.nodeid, self._domain.cells.periodicnid,
                            self._domain.nodes.periodicid,
                            self._domain.nodes.oldname, self._domain.halos.centvol, self._domain.cells.shift,
                            self.gradcellx,
                            self.gradcelly, self.gradcellz)

    # The limiter depend on hc value
    self._barthlimiter(self.cell, self.ghost, self.halo, self.gradcellx, self.gradcelly, self.gradcellz,
                           self.psi, self._domain.faces.cellid, self._domain.cells.faceid, self._domain.faces.name,
                           self._domain.faces.halofid, self._domain.cells.center, self._domain.faces.center, self.domain.faces.ghost_id)

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
                            self._domain.periodicboundaryfaces, self.domain.faces.ghost_id)

  def update_ghost_value(self):
    for BC in self._BCs.values():
      BC.func_ghost(BC.BCvalueface, self.ghost, self._domain.faces.cellid,
                     BC.BCfaces,
                     BC.constNH, self._domain.faces.dist_ortho, self._domain.faces.ghost_id)
      BC.func_haloghost(BC.BCvaluehalo, self.haloghost, self._domain.nodes.haloghostid,
                         self._domain.ghost.ext_info_int, self._domain.ghost.ext_info_flt, BC.BCtypeindex, self._domain.halonodes, BC.constNHNode)

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