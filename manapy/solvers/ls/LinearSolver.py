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
import manapy.solvers.ls.ls_diamond as ls_diamond
import manapy.solvers.ls.ls_fv as ls_fv
from manapy.backends.types import FLOAT_TYPE, np_float_type
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

class LinearSolver:
  _parameters = [('scheme', 'str', 'diamond', 'diamond',
                  'scheme: diamond, fv (orthogonal) or fv_corrected (non-orthogonal corrected).'),
                 ('reordering', 'bool', False, False,
                  'reordering the matrix only in serial case.')
                 ]

  SolverPetsc = "petsc"
  SolverMumps = "mumps"
  SolverScipy = "scipy"
  SolverGinkgo = "ginkgo"
  SolverGinkgoDist = "ginkgo_dist"

  def __init__(self,
               domain: Domain,
               var: Variable,
               comm: MPI.Comm = MPI.COMM_WORLD,
               scheme: str = "diamond",
               reordering: bool = False,
               solver_name: str = None,
               non_orthogonal_corrections: int = 2,
               non_orthogonal_limiter: float = 1.0,
     ):

    if type(self) is LinearSolver:
      raise TypeError("Base class cannot be instantiated directly")
    if solver_name not in [LinearSolver.SolverPetsc, LinearSolver.SolverMumps, LinearSolver.SolverScipy, LinearSolver.SolverGinkgo, LinearSolver.SolverGinkgoDist]:
      raise Exception("Unexpected solver type")

    self.comm = comm


    if self.comm.Get_rank() == 0:
      print("SetUp the Linear system ...")


    self.scheme = scheme
    # Cell-centred finite-volume schemes (vs the vertex-based "diamond"):
    #   "fv"           -> orthogonal two-point Laplacian, NO correction.
    #   "fv_corrected" -> same matrix + explicit non-orthogonal correction
    #                     (deferred-correction loop, `non_orthogonal_corrections`
    #                     cycles). This is the OpenFOAM Gauss-linear-corrected one.
    # Both share the same assembly (fv_schemes); only "fv_corrected" runs the
    # correction loop, so `non_orthogonal_corrections` is ignored for plain "fv".
    self.fv_schemes = ("fv", "fv_corrected")
    self.non_orthogonal_corrections = non_orthogonal_corrections if scheme == "fv_corrected" else 0
    # Non-orthogonal correction blend psi in [0, 1] (OpenFOAM-like "limited"):
    #   1.0 -> full correction (fv_corrected), 0.0 -> none (= fv),
    #   0.333/0.5 -> partial, for robustness on highly non-orthogonal meshes.
    self.non_orthogonal_limiter = float(non_orthogonal_limiter)
    self.verbose = False

    self.var = var
    self.domain = domain
    self.dim = self.domain.dim
    ls_compute.setup(self.dim)                 # shared dim-agnostic helpers
    if self.scheme in self.fv_schemes:         # compile only the scheme in use
      ls_fv.setup(self.dim)
    else:                                       # diamond
      ls_diamond.setup(self.dim)
    self.mpi_precision = MPI.FLOAT if FLOAT_TYPE == "float32" else MPI.DOUBLE

    # GPU backend: matrix/RHS/gradient assembly run on numba.cuda kernels
    # (cuda_ls_compute) instead of the numba CPU kernels (ls_compute). 2D only.
    self._gpu = getattr(self.domain.backend, "name", "cpu") == "gpu"
    if self._gpu:
      if self.dim != 2:
        raise NotImplementedError("LinearSolver GPU kernels are 2D-only for now")
      import manapy.solvers.ls.cuda_ls_compute as gls
      from manapy.backends.gpu import GPUArray
      self._gls = gls
      self._GPUArray = GPUArray

    # Backend

    self.localsize = self.domain.nbcells
    self.globalsize = self.comm.allreduce(self.localsize, op=MPI.SUM)
    self.domain.globalsize = self.globalsize

    self.sendcounts1 = self.comm.gather(self.localsize, root=0)
    if self.comm.Get_rank() == 0:
      self.sendcounts1 = np.array(self.sendcounts1, dtype=np.int32)
    self.x1converted = np.zeros(self.globalsize, dtype=FLOAT_TYPE)

    # Pbordnode/Pbordface : ecrits par les kernels dirichlet/neumann (device sous
    # GPU) puis lus par get_rhs/get_triplet/gradient -> alloues sur le backend.
    _be = self.domain.backend
    self.domain.Pbordnode = _be.zeros(self.domain.nbnodes, FLOAT_TYPE)
    self.domain.Pbordface = _be.zeros(self.domain.nbfaces, FLOAT_TYPE)
    self.domain.Ibordnode = _be.zeros(self.domain.nbnodes, FLOAT_TYPE)
    self.domain.Ibordface = _be.zeros(self.domain.nbfaces, FLOAT_TYPE)

    matrixinnerfaces = np.concatenate(
      [self.domain.innerfaces, self.domain.periodicinfaces, self.domain.periodicupperfaces])
    if self.dim == 3:
      matrixinnerfaces = np.concatenate([matrixinnerfaces, self.domain.periodicfrontfaces])
    self.matrixinnerfaces = np.sort(matrixinnerfaces)

    if scheme in self.fv_schemes:
      if self._gpu:
        raise NotImplementedError("FV-like Laplacian assembly is CPU-only for now")
      self._compute_P_gradient = None
      self._get_triplet = ls_fv.get_triplet_fv
      self.dataSize = ls_fv.compute_fv_matrix_size(
        self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
      self._row = np.zeros(self.dataSize, dtype=np.int32)
      self._col = np.zeros(self.dataSize, dtype=np.int32)
      self._data = np.zeros(self.dataSize, dtype=FLOAT_TYPE)

    elif scheme == "diamond":
      if self.dim == 2:
        # "pure map" gradient: one grid-stride body wrapped for CPU or GPU.
        self._compute_P_gradient = self.domain.backend.make_gridstride_kernel(
          ls_diamond._gs_compute_P_gradient_2d_diamond, size_arg=(21, 22, 23, 24, 25))
        # matrix assembly needs atomics -> hand-written GPU kernel, CPU njit.
        if self._gpu:
          self._get_triplet = self._gls.get_kernel_get_triplet_2d()
          _matrix_size = self._gls.get_kernel_compute_2dmatrix_size()
        else:
          self._get_triplet = ls_diamond.get_triplet_2d
          _matrix_size = ls_diamond.compute_2dmatrix_size
        self.dataSize = _matrix_size(self.domain.faces.nodeid,
                                                          self.domain.nodes.cellid,
                                                          self.domain.nodes.halonid,
                                                          self.domain.nodes.periodicid,
                                                          self.domain.nodes.ghostid,
                                                          self.domain.nodes.haloghostid,
                                                          self.domain.nodes.oldname,
                                                          self.var.BCdirichlet,
                                                          self.matrixinnerfaces,
                                                          self.domain.halofaces,
                                                          self.var.dirichletfaces)
      elif self.dim == 3:
        self._compute_P_gradient = ls_diamond.compute_P_gradient_3d_diamond
        self._get_triplet = ls_diamond.get_triplet_3d
        self.dataSize = ls_diamond.compute_3dmatrix_size(self.domain.faces.nodeid,
                                                          self.domain.nodes.cellid,
                                                          self.domain.nodes.halonid,
                                                          self.domain.nodes.periodicid,
                                                          self.domain.nodes.ghostid,
                                                          self.domain.nodes.haloghostid,
                                                          self.domain.nodes.oldname,
                                                          self.var.BCdirichlet,
                                                          self.matrixinnerfaces,
                                                          self.domain.halofaces,
                                                          self.var.dirichletfaces)

      # Alloues dans la memoire du backend : ecrits par le kernel d'assemblage
      # (get_triplet) puis lus par le solveur, cote device sous GPU.
      _be = self.domain.backend
      self._row = _be.zeros(self.dataSize, np.int32)
      self._col = _be.zeros(self.dataSize, np.int32)
      self._data = _be.zeros(self.dataSize, FLOAT_TYPE)
      if self._gpu:
        self.matrixinnerfaces = _be.asarray(self.matrixinnerfaces, np.int32)

    else:
      raise ValueError("unknown linear solver scheme "
                       f"'{scheme}'; choose diamond or fv")

    _glob = solver_name in [LinearSolver.SolverScipy, LinearSolver.SolverMumps,
                            LinearSolver.SolverGinkgo]
    if self._gpu:
      # 2D-only GPU RHS kernels (glob for centralized solvers, loc otherwise).
      self._get_rhs = (self._gls.get_kernel_get_rhs_glob_2d() if _glob
                       else self._gls.get_kernel_get_rhs_loc_2d())
    elif scheme in self.fv_schemes:
      self._get_rhs = (ls_fv.get_rhs_fv_glob if _glob
                       else ls_fv.get_rhs_fv_loc)
      self._get_rhs_correction = (ls_fv.get_rhs_fv_correction_glob if _glob
                                  else ls_fv.get_rhs_fv_correction_loc)
    elif _glob:
      self._get_rhs = ls_diamond.get_rhs_glob_2d if self.dim == 2 else ls_diamond.get_rhs_glob_3d
    else:
      self._get_rhs = ls_diamond.get_rhs_loc_2d if self.dim == 2 else ls_diamond.get_rhs_loc_3d

    # "pure map" kernels: single grid-stride body wrapped for CPU or GPU.
    _be = self.domain.backend
    self.convert_solution = _be.make_gridstride_kernel(
      ls_compute._gs_convert_solution, size_arg=lambda a: a[3])  # b0Size
    self._rhs_value_dirichlet_node = _be.make_gridstride_kernel(
      ls_compute._gs_rhs_value_dirichlet_node, size_arg=1)       # nodes
    self._rhs_value_dirichlet_face = _be.make_gridstride_kernel(
      ls_compute._gs_rhs_value_dirichlet_face, size_arg=1)       # faces
    self._set_scalar_at = _be.make_gridstride_kernel(
      ls_compute._gs_set_scalar_at, size_arg=1)                  # idx
    self._prepare_bc_update_tables()

  def _index_array(self, values):
    values = np.asarray(values, dtype=np.int32)
    if self._gpu:
      return self.domain.backend.asarray(values, np.int32)
    return values

  def _prepare_bc_update_tables(self):
    oldname = self.domain.backend.to_host(self.domain.nodes.oldname) if self._gpu else self.domain.nodes.oldname
    self._bc_update_tables = []
    for BC in self.var.BCs.values():
      if BC is None:
        continue
      if BC.BCtype == "dirichlet":
        self._bc_update_tables.append((
          "dirichlet",
          BC,
          self._index_array(BC.BCfaces),
          self._index_array(np.where(oldname == BC.BCtypeindex)[0]),
        ))
      elif BC.BCtype == "neumann":
        self._bc_update_tables.append((
          "neumann",
          BC,
          None,
          self._index_array(np.where(oldname == BC.BCtypeindex)[0]),
        ))

  def assembly(self):
    if self.scheme in self.fv_schemes:
      self._get_triplet(self.domain.faces.cellid, self.domain.faces.fv_coeff,
                        self.domain.halos.halosext, self.domain.cells.volume,
                        self.domain.cells.loctoglob, self.domain.faces.halofid,
                        self._data, self._row, self._col, self.matrixinnerfaces,
                        self.domain.halofaces, self.var.dirichletfaces)

      self._get_rhs(self.domain.faces.cellid, self.domain.faces.fv_coeff,
                    self.domain.cells.volume, self.domain.cells.loctoglob,
                    self.domain.Pbordface, self.rhs0, self.var.dirichletfaces)
    else:
      self._get_triplet(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.vertex,
                        self.domain.faces.halofid,
                        self.domain.halos.halosext, self.domain.nodes.oldname, self.domain.cells.volume,
                        self.domain.nodes.cellid, self.domain.cells.center, self.domain.halos.centvol,
                        self.domain.nodes.halonid, self.domain.nodes.periodicid,
                        self.domain.ghost.info_flt, self.domain.ghost.ext_info_flt, self.domain.ghost.info_int, self.domain.ghost.ext_info_int,
                        self.domain.nodes.ghostid, self.domain.nodes.haloghostid,
                        self.domain.faces.airDiamond,
                        self.domain.nodes.lambda_x, self.domain.nodes.lambda_y, self.domain.nodes.lambda_z,
                        self.domain.nodes.number, self.domain.nodes.R_x,
                        self.domain.nodes.R_y, self.domain.nodes.R_z, self.domain.faces.param1, self.domain.faces.param2,
                        self.domain.faces.param3,
                        self.domain.faces.param4, self.domain.cells.shift, self.localsize, self.domain.cells.loctoglob,
                        self.var.BCdirichlet, self._data,
                        self._row, self._col, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)

      self._get_rhs(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                    self.domain.cells.volume, self.domain.nodes.ghostid, self.domain.cells.loctoglob,
                    self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3,
                    self.domain.faces.param4, self.domain.Pbordnode, self.domain.Pbordface,
                    self.rhs0, self.var.BCdirichlet,
                    self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)

  def update_ghost_values(self):
    for kind, BC, faces, nodes in self._bc_update_tables:
      if kind == "dirichlet":
        self._rhs_value_dirichlet_face(self.domain.Pbordface, faces, BC.BCvalueface)
        self._rhs_value_dirichlet_node(self.domain.Pbordnode, nodes, BC.BCvaluenode)

      elif kind == "neumann":
        # Pbordnode[neumann_nodes] = 1, via kernel (Pbordnode est sur le backend).
        self._set_scalar_at(self.domain.Pbordnode, nodes, 1.0)

  def compute_Sol_gradient(self):
    if self.scheme in self.fv_schemes:
      # The corrected face gradient needs the full cell gradient (its tangential
      # part); the two-point term only supplies the normal component.
      self.var.update_halo_value()
      self.var.update_ghost_value()
      self.var.compute_cell_gradient()
      ls_fv.compute_P_gradient_fv(
        self.var.cell, self.var.halo, self.domain.faces.cellid,
        self.domain.faces.name, self.domain.faces.normal, self.domain.faces.center,
        self.domain.faces.halofid, self.domain.cells.center,
        self.domain.halos.centvol, self.domain.cells.shift, self.domain.Pbordface,
        self.var.gradcellx, self.var.gradcelly, self.var.gradcellz,
        self.var.gradhalocellx, self.var.gradhalocelly, self.var.gradhalocellz,
        self.domain.faces.fv_weight_left,
        self.var.gradfacex, self.var.gradfacey, self.var.gradfacez,
        self.domain.innerfaces, self.domain.halofaces, self.var.neumannfaces,
        self.var.dirichletfaces, self.domain.periodicboundaryfaces)
    else:
      self._compute_P_gradient(self.var.cell, self.var.ghost, self.var.halo, self.var.node, self.domain.faces.cellid,
                               self.domain.faces.nodeid,
                               self.domain.faces.halofid, self.domain.nodes.oldname, self.domain.faces.airDiamond,
                               self.domain.faces.f_1, self.domain.faces.f_2, self.domain.faces.f_3, self.domain.faces.f_4,
                               self.domain.faces.normal, self.domain.cells.shift, self.domain.Pbordnode,
                               self.domain.Pbordface,
                               self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.var.BCdirichlet,
                               self.domain.innerfaces, self.domain.halofaces, self.var.neumannfaces,
                               self.var.dirichletfaces, self.domain.periodicboundaryfaces)


  def fv_correction_rhs(self, global_rhs=True):
    """Build the explicit non-orthogonal correction source for the current solution."""
    self.var.update_halo_value()
    self.var.update_ghost_value()
    self.var.compute_cell_gradient()

    size = self.globalsize if global_rhs else self.localsize
    corr = np.zeros(size, dtype=FLOAT_TYPE)
    self._get_rhs_correction(
      self.domain.faces.cellid, self.domain.faces.halofid, self.domain.cells.volume,
      self.domain.cells.loctoglob, self.domain.faces.fv_corrx,
      self.domain.faces.fv_corry, self.domain.faces.fv_corrz,
      self.domain.faces.fv_weight_left, self.var.gradcellx,
      self.var.gradcelly, self.var.gradcellz,
      self.var.gradhalocellx, self.var.gradhalocelly, self.var.gradhalocellz,
      corr, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces,
      self.domain.periodicboundaryfaces)
    # "limited" blend: scale the explicit non-orthogonal correction by psi.
    if self.non_orthogonal_limiter != 1.0:
      corr *= self.non_orthogonal_limiter
    if global_rhs:
      return self.comm.reduce(corr, op=MPI.SUM, root=0)
    return corr

  def fv_corrected_rhs(self, base_rhs, user_rhs=None, global_rhs=True):
    rhs = base_rhs.copy() if base_rhs is not None else None
    if user_rhs is not None:
      rhs = user_rhs.copy() if rhs is None else rhs + user_rhs
    corr = self.fv_correction_rhs(global_rhs=global_rhs)
    if global_rhs:
      if self.comm.Get_rank() == 0:
        rhs = corr if rhs is None else rhs + corr
      return rhs
    return corr if rhs is None else rhs + corr

  def reordering_matrix(self):
    matrix = csr_matrix((self._data, (self._row, self._col)))
    # Compute the reverse Cuthill-Mckee ordering
    self.perm = reverse_cuthill_mckee(matrix, symmetric_mode=False)
    matrix = matrix[:, self.perm][self.perm, :]
    ## Convert the reordered matrix back to AIJ format
    self._row, self._col = matrix.nonzero()
    self._data = matrix.data
    self.rhs0 = self.rhs0[self.perm]








