#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:54:55 2022

@author: kissami
"""
import numpy as np
from mpi4py import MPI
from manapy.core.Variable import Variable
from manapy.domain import Domain
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.backends.types import FLOAT_TYPE, np_float_type
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

class LinearSolver:
  _parameters = [('scheme', 'str', 'diamond', 'fv4',
                  'scheme diamond or fv4.'),
                 ('reordering', 'bool', False, False,
                  'reordering the matrix only in serial case.')
                 ]

  SolverPetsc = "petsc"
  SolverMumps = "mumps"
  SolverScipy = "scipy"

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Comm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               solver_name: str = None,
     ):

    if type(self) is LinearSolver:
      raise TypeError("Base class cannot be instantiated directly")
    if solver_name not in [LinearSolver.SolverPetsc, LinearSolver.SolverMumps, LinearSolver.SolverScipy]:
      raise Exception("Unexpected solver type")

    self.comm = comm


    if self.comm.Get_rank() == 0:
      print("SetUp the Linear system ...")


    self.scheme = scheme
    self.verbose = False

    self.var = var
    self.domain = domain
    self.dim = self.domain.dim
    self.mpi_precision = MPI.FLOAT if FLOAT_TYPE == "float32" else MPI.DOUBLE

    # Backend

    self.localsize = self.domain.nbcells
    self.globalsize = self.comm.allreduce(self.localsize, op=MPI.SUM)
    self.domain.globalsize = self.globalsize

    self.sendcounts1 = self.comm.gather(self.localsize, root=0)
    if self.comm.Get_rank() == 0:
      self.sendcounts1 = np.array(self.sendcounts1, dtype=np.int32)
    self.x1converted = np.zeros(self.globalsize, dtype=FLOAT_TYPE)

    self.domain.Pbordnode = np.zeros(self.domain.nbnodes, dtype=FLOAT_TYPE)
    self.domain.Pbordface = np.zeros(self.domain.nbfaces, dtype=FLOAT_TYPE)

    self.domain.Ibordnode = np.zeros(self.domain.nbnodes, dtype=FLOAT_TYPE)
    self.domain.Ibordface = np.zeros(self.domain.nbfaces, dtype=FLOAT_TYPE)

    matrixinnerfaces = np.concatenate(
      [self.domain.innerfaces, self.domain.periodicinfaces, self.domain.periodicupperfaces])
    if self.dim == 3:
      matrixinnerfaces = np.concatenate([matrixinnerfaces, self.domain.periodicfrontfaces])
    self.matrixinnerfaces = np.sort(matrixinnerfaces)

    if scheme == "fv4":
      self._compute_P_gradient = ls_compute.compute_P_gradient_2d_FV4
      sizeM = 4 * len(matrixinnerfaces) + len(self.var.dirichletfaces) + 2 * len(self.domain.halofaces)
      self._row = np.zeros(sizeM, dtype=np.int32)
      self._col = np.zeros(sizeM, dtype=np.int32)
      self._data = np.zeros(sizeM, dtype=FLOAT_TYPE)

    elif scheme == "diamond":
      if self.dim == 2:
        self._compute_P_gradient = ls_compute.compute_P_gradient_2d_diamond
        self._get_triplet = ls_compute.get_triplet_2d
        self.dataSize = ls_compute.compute_2dmatrix_size(self.domain.faces.nodeid,
                                                          self.domain.faces.halofid,
                                                          self.domain.nodes.cellid,
                                                          self.domain.nodes.halonid,
                                                          self.domain.nodes.periodicid,
                                                          self.domain.nodes.ghostcenter_info,
                                                          self.domain.nodes.haloghostid,
                                                          self.domain.nodes.oldname,
                                                          self.var.BCdirichlet,
                                                          self.matrixinnerfaces,
                                                          self.domain.halofaces,
                                                          self.var.dirichletfaces)
      elif self.dim == 3:
        self._compute_P_gradient = ls_compute.compute_P_gradient_3d_diamond
        self._get_triplet = ls_compute.get_triplet_3d
        self.dataSize = ls_compute.compute_3dmatrix_size(self.domain.faces.nodeid,
                                                          self.domain.faces.halofid,
                                                          self.domain.nodes.cellid,
                                                          self.domain.nodes.halonid,
                                                          self.domain.nodes.periodicid,
                                                          self.domain.nodes.ghostcenter_info,
                                                          self.domain.nodes.haloghostid,
                                                          self.domain.nodes.oldname,
                                                          self.var.BCdirichlet,
                                                          self.matrixinnerfaces,
                                                          self.domain.halofaces,
                                                          self.var.dirichletfaces)

      self._row = np.zeros(self.dataSize, dtype=np.int32)
      self._col = np.zeros(self.dataSize, dtype=np.int32)
      self._data = np.zeros(self.dataSize, dtype=FLOAT_TYPE)

    if solver_name in [LinearSolver.SolverScipy, LinearSolver.SolverMumps]:
      self._get_rhs = ls_compute.get_rhs_glob_2d
      if self.dim == 3:
        self._get_rhs = ls_compute.get_rhs_glob_3d
    elif solver_name == LinearSolver.SolverPetsc:
      self._get_rhs = ls_compute.get_rhs_loc_2d
      if self.dim == 3:
        self._get_rhs = ls_compute.get_rhs_loc_3d

    self.convert_solution = ls_compute.convert_solution

  def assembly(self):
    self._get_triplet(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.vertex,
                      self.domain.faces.halofid,
                      self.domain.halos.halosext, self.domain.nodes.oldname, self.domain.cells.volume,
                      self.domain.nodes.cellid, self.domain.cells.center, self.domain.halos.centvol,
                      self.domain.nodes.halonid, self.domain.nodes.periodicid,
                      self.domain.nodes.ghostcenter, self.domain.nodes.ghostcenter_info, self.domain.nodes.haloghostid, self.domain.nodes.haloghostcenter, self.domain.nodes.haloghostcenter_info, self.domain.faces.airDiamond,
                      self.domain.nodes.lambda_x, self.domain.nodes.lambda_y, self.domain.nodes.lambda_z,
                      self.domain.nodes.number, self.domain.nodes.R_x,
                      self.domain.nodes.R_y, self.domain.nodes.R_z, self.domain.faces.param1, self.domain.faces.param2,
                      self.domain.faces.param3,
                      self.domain.faces.param4, self.domain.cells.shift, self.localsize, self.domain.cells.loctoglob,
                      self.var.BCdirichlet, self._data,
                      self._row, self._col, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)

    self._get_rhs(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                  self.domain.cells.volume, self.domain.nodes.ghostcenter_info, self.domain.cells.loctoglob,
                  self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3,
                  self.domain.faces.param4, self.domain.Pbordnode, self.domain.Pbordface,
                  self.rhs0, self.var.BCdirichlet, self.domain.faces.ghostcenter,
                  self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)

  def update_ghost_values(self):
    for BC in self.var.BCs.values():
      if BC.BCtype == "dirichlet":
        ls_compute.rhs_value_dirichlet_face(self.domain.Pbordface, np.asarray(BC.BCfaces, dtype=np.int32), BC.BCvalueface)
        ls_compute.rhs_value_dirichlet_node(self.domain.Pbordnode,
                                 np.where(self.domain.nodes.oldname == BC.BCtypeindex)[0].astype(np.int32),
                                 BC.BCvaluenode)

      elif BC.BCtype == "neumann":
        for i in np.where(self.domain.nodes.oldname == BC.BCtypeindex)[0]:
          self.domain.Pbordnode[i] = 1.

  def compute_Sol_gradient(self):
    self._compute_P_gradient(self.var.cell, self.var.ghost, self.var.halo, self.var.node, self.domain.faces.cellid,
                             self.domain.faces.nodeid, self.domain.faces.ghostcenter,
                             self.domain.faces.halofid, self.domain.cells.center,
                             self.domain.halos.centvol, self.domain.nodes.oldname, self.domain.faces.airDiamond,
                             self.domain.faces.f_1, self.domain.faces.f_2, self.domain.faces.f_3, self.domain.faces.f_4,
                             self.domain.faces.normal, self.domain.cells.shift, self.domain.Pbordnode,
                             self.domain.Pbordface,
                             self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.var.BCdirichlet,
                             self.domain.innerfaces, self.domain.halofaces, self.var.neumannfaces,
                             self.var.dirichletfaces, self.domain.periodicboundaryfaces)

  def reordering_matrix(self):
    matrix = csr_matrix((self._data, (self._row, self._col)))
    # Compute the reverse Cuthill-Mckee ordering
    self.perm = reverse_cuthill_mckee(matrix, symmetric_mode=False)
    matrix = matrix[:, self.perm][self.perm, :]
    ## Convert the reordered matrix back to AIJ format
    self._row, self._col = matrix.nonzero()
    self._data = matrix.data
    self.rhs0 = self.rhs0[self.perm]








