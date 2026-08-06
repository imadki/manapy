#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping, Optional, Union

from manapy.backends import ManapyArray
from manapy.backends.ManapyArray import Device, cp
from manapy.domain import Domain
from manapy.boundary.Boundary import Boundary
from manapy.compute import VariableCompute


BoundaryValue = Union[Real, Callable[[Any, Any, Any], Any]]


@dataclass(frozen=True)
class BoundarySetup:
  """Boundary groups used by reconstruction and linear-system kernels."""

  neumann_faces: ManapyArray
  neumann_indices: ManapyArray
  dirichlet_faces: ManapyArray
  dirichlet_indices: ManapyArray
  neumann_nh_faces: ManapyArray
  neumann_nh_indices: ManapyArray
  boundaries: dict[str, Boundary]

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


  def __init__(
      self,
      domain: Domain,
      BC: Optional[Mapping[str, str]] = None,
      values_dict: Optional[Mapping[str, BoundaryValue]] = None,
      name: Optional[str] = None,
      limiter: str = "barth",
  ) -> None:
    if domain is None:
      raise ValueError("domain must be given")

    self._limiter = str(limiter).lower() if limiter else 'barth'
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
    boundary_setup = self._update_boundaries(BC, self._values)
    self.neumannfaces = boundary_setup.neumann_faces
    self.BCneumann = boundary_setup.neumann_indices
    self.dirichletfaces = boundary_setup.dirichlet_faces
    self.BCdirichlet = boundary_setup.dirichlet_indices
    self.neumannNHfaces = boundary_setup.neumann_nh_faces
    self.BCneumannNH = boundary_setup.neumann_nh_indices
    self._BCs = boundary_setup.boundaries

    # A "slip" wall is a coupled (vector) BC: it needs every velocity component
    # together. Variables carrying a slip BC auto-register on the domain in
    # creation order (u, v[, w]), so update_ghost_value() can apply the coupled
    # slip automatically without the user wiring update_slip_ghost() by hand.
    self._has_slip = any(boundary.is_vector for boundary in self._BCs.values())
    if self._has_slip:
      if self._domain.slip_velocity is None:
        self._domain.slip_velocity = []
      self._domain.slip_velocity.append(self)
      if self._domain.rank == 0:
        print("WARNING: 'slip' boundary detected (coupled vector BC).\n"
              "         Create the velocity components in spatial order "
              "(u, v[, w]), each with the slip BC;\n"
              "         only velocity components must carry 'slip'. They "
              "auto-register and the reflection\n"
              "         is applied by update_ghost_value() in that order.")

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

  def add_term(self, name: str) -> None:
    # Alloue dans la memoire du backend (device sous GPU, host sous CPU).
    self.__dict__[name] = ManapyArray.zeros(self._nbcells, self.config.float_dtype, self.config.device)

  def _fill_bc_values(
      self,
      value: BoundaryValue,
      bcfaces: np.ndarray,
      bctypeindex: int,
      valueface: np.ndarray,
      valuenode: np.ndarray,
      valuehalo: np.ndarray,
  ) -> None:
    """Fill the face / node / haloghost boundary arrays from a prescribed value.

    `value` may be a constant (int/float) or a callable ``lambda x, y, z``.
    Shared by the dirichlet and neumannNH branches of `_update_boundaries`.
    """
    node_boundary_indices = self._domain.nodes.oldname.cpu_r()
    bcnodes = np.where(node_boundary_indices == bctypeindex)[0]

    # Flatten the ragged (node -> haloghost ids) map for the boundary nodes:
    # the last column holds the per-node count, the others the ghost ids.
    haloghostid = self._domain.nodes.haloghostid.cpu_r()
    if len(bcnodes):
      maxg = haloghostid.shape[1] - 1
      counts = haloghostid[bcnodes, -1]
      mask = np.arange(maxg)[None, :] < counts[:, None]
      ghost_ids = haloghostid[bcnodes][:, :maxg][mask]
    else:
      ghost_ids = np.empty(0, dtype=haloghostid.dtype)

    fc = self._domain.faces.center.cpu_r()
    vx = self._domain.nodes.vertex.cpu_r()
    gfc = self._domain.ghost.ext_info_flt.cpu_r()

    if callable(value):
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

    elif isinstance(value, Real):
      valueface[bcfaces] = value
      valuenode[bcnodes] = value
      if len(ghost_ids):
        valuehalo[ghost_ids] = value

    else:
      raise ValueError("BC value must be a number or a callable lambda(x, y, z)")

  def _update_boundaries(
      self,
      requested_bcs: Optional[Mapping[str, str]],
      values_dict: Optional[Mapping[str, BoundaryValue]],
  ) -> BoundarySetup:
    """Build every domain boundary and the face groups used by FV kernels."""
    prescribed_face_values = np.zeros(
      self._domain.nbfaces, dtype=self.config.float_dtype
    )
    prescribed_node_values = np.zeros(
      self._domain.nbnodes, dtype=self.config.float_dtype
    )
    prescribed_halo_values = np.zeros(
      self._domain.halos.sizehaloghost, dtype=self.config.float_dtype
    )

    neumann_faces: list[int] = []
    neumann_indices: list[int] = []
    dirichlet_faces: list[int] = []
    dirichlet_indices: list[int] = []
    neumann_nh_faces: list[int] = []
    neumann_nh_indices: list[int] = []

    if self._dim == 2:
      locations = ("in", "out", "bottom", "upper")
    else:
      locations = ("in", "out", "bottom", "upper", "front", "back")

    if requested_bcs is not None:
      for location in requested_bcs:
        if location not in locations:
          raise ValueError(f"Unknown boundary location: {location}")

    boundaries: dict[str, Boundary] = {}
    dirichlet_boundaries: list[Boundary] = []
    neumann_nh_boundaries: list[Boundary] = []

    for location in locations:
      domain_bc_type = self._domain.BCs[location][0]
      domain_bc_index = self._domain.BCs[location][1]

      if requested_bcs is None or location not in requested_bcs:
        bc_type = domain_bc_type
      else:
        bc_type = requested_bcs[location]

      if domain_bc_type == "periodic" and bc_type != "periodic":
        raise ValueError(f"Boundary {location} must be periodic")
      if domain_bc_type != "periodic" and bc_type == "periodic":
        raise ValueError(f"Boundary {location} is not periodic in the domain")

      boundary = Boundary(
        BCtype=bc_type,
        BCloc=location,
        BCtypeindex=domain_bc_index,
        domain=self._domain,
        default_value_face=self.cell,
        default_value_node=self.node,
        default_value_halo=self.halo,
      )
      boundaries[location] = boundary
      boundary_faces = boundary.BCfaces.cpu_r()

      if bc_type == "dirichlet":
        if values_dict is None or location not in values_dict:
          raise ValueError(
            f"Value of dirichlet BC for {location} faces must be given"
          )
        dirichlet_indices.append(boundary.BCtypeindex)
        dirichlet_faces.extend(boundary_faces.tolist())
        dirichlet_boundaries.append(boundary)
        self._fill_bc_values(
          values_dict[location],
          boundary_faces,
          boundary.BCtypeindex,
          prescribed_face_values,
          prescribed_node_values,
          prescribed_halo_values,
        )
      elif bc_type == "neumannNH":
        if values_dict is None or location not in values_dict:
          raise ValueError(
            f"Value of neumannNH BC for {location} faces must be given"
          )
        neumann_nh_indices.append(boundary.BCtypeindex)
        neumann_nh_faces.extend(boundary_faces.tolist())
        neumann_nh_boundaries.append(boundary)
        self._fill_bc_values(
          values_dict[location],
          boundary_faces,
          boundary.BCtypeindex,
          prescribed_face_values,
          prescribed_node_values,
          prescribed_halo_values,
        )
      elif bc_type == "neumann":
        neumann_indices.append(boundary.BCtypeindex)
        neumann_faces.extend(boundary_faces.tolist())
      elif bc_type == "slip":
        neumann_indices.append(boundary.BCtypeindex)
        neumann_faces.extend(boundary_faces.tolist())
      elif bc_type == "nonslip":
        neumann_indices.append(boundary.BCtypeindex)
        neumann_faces.extend(boundary_faces.tolist())
      elif bc_type == "periodic":
        pass
      else:
        # Boundary validates the type before this point. Keeping this branch
        # makes the bookkeeping decision explicit if new types are added.
        raise ValueError(f"Boundary type {bc_type} has no face-group rule")

    prescribed_faces = ManapyArray.array(
      prescribed_face_values,
      dtype=self.config.float_dtype,
      device=self.config.device,
    )
    prescribed_nodes = ManapyArray.array(
      prescribed_node_values,
      dtype=self.config.float_dtype,
      device=self.config.device,
    )
    prescribed_halos = ManapyArray.array(
      prescribed_halo_values,
      dtype=self.config.float_dtype,
      device=self.config.device,
    )

    for boundary in dirichlet_boundaries:
      boundary.BCvalueface = prescribed_faces
      boundary.BCvaluenode = prescribed_nodes
      boundary.BCvaluehalo = prescribed_halos

    for boundary in neumann_nh_boundaries:
      boundary.constNH = prescribed_faces
      boundary.constNHGhost = prescribed_halos

    neumann_faces.sort()
    dirichlet_faces.sort()

    return BoundarySetup(
      neumann_faces=ManapyArray.array(
        neumann_faces, dtype=self.config.int_dtype, device=self.config.device
      ),
      neumann_indices=ManapyArray.array(
        neumann_indices, dtype=self.config.int_dtype, device=self.config.device
      ),
      dirichlet_faces=ManapyArray.array(
        dirichlet_faces, dtype=self.config.int_dtype, device=self.config.device
      ),
      dirichlet_indices=ManapyArray.array(
        dirichlet_indices, dtype=self.config.int_dtype, device=self.config.device
      ),
      neumann_nh_faces=ManapyArray.array(
        neumann_nh_faces, dtype=self.config.int_dtype, device=self.config.device
      ),
      neumann_nh_indices=ManapyArray.array(
        neumann_nh_indices, dtype=self.config.int_dtype, device=self.config.device
      ),
      boundaries=boundaries,
    )

  def update_halo_value(self) -> None:
    # update the halo values
    self.domain.halo_comm.exchange(self.cell, recv_buffer=self.halo)


  def interpolate_facetocell(self) -> None:
    self._facetocell(self.face, self.cell, self._domain.cells.faceid, self._dim)

  def interpolate_celltoface(self) -> None:
    self._celltoface(self.cell, self.face, self.ghost, self.halo, self._domain.faces.cellid,
                         self._domain.faces.halofid,
                         self._domain.innerfaces, self._domain.boundaryfaces, self._domain.halofaces)

  def interpolate_celltonode(self) -> None:
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

  def compute_cell_gradient(self) -> None:
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

  def compute_face_gradient(self) -> None:
    self._face_gradient(self.cell, self.ghost, self.halo, self.node, self._domain.faces.cellid,
                            self._domain.faces.nodeid,
                            self._domain.faces.halofid, self._domain.faces.airDiamond,
                            self._domain.faces.normal, self._domain.faces.f_1, self._domain.faces.f_2,
                            self._domain.faces.f_3, self._domain.faces.f_4,
                            self.gradfacex, self.gradfacey, self.gradfacez, self._domain.innerfaces,
                            self._domain.halofaces, self.dirichletfaces, self.neumannfaces,
                            self._domain.periodicboundaryfaces)

  def update_ghost_value(self) -> None:
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
      Boundary.update_slip_ghost(self._domain.slip_velocity)

  def norml2(
      self, exact: Union[ManapyArray, np.ndarray], order: Optional[int] = None
  ) -> float:
    """Volume-weighted relative error against an exact solution.

    `exact` may be a ManapyArray, a numpy array or any array-like. Everything
    is read on the config device and reduced there, so on the GPU path only the
    scalar result crosses the bus -- the per-element indexing this used to do
    would have forced a sync per cell.

    Read-only throughout: `cell` and `cells.volume` keep their copy on the
    other device valid, so calling this does not invalidate anything.
    """
    if order is None:
      order = 1
    assert self._nbcells == len(exact), 'exact solution must have length of cells'

    if self.config.device == Device.CUDA:
      xp = cp
      cell = self.cell.gpu_r()
      volume = self._domain.cells.volume.gpu_r()
      exact = exact.gpu_r() if isinstance(exact, ManapyArray) else cp.asarray(exact)
    else:
      xp = np
      cell = self.cell.cpu_r()
      volume = self._domain.cells.volume.cpu_r()
      exact = exact.cpu_r() if isinstance(exact, ManapyArray) else np.asarray(exact)

    error = xp.abs(cell - exact) * volume
    ex = xp.abs(exact) * volume

    # float() also forces the CuPy reduction to land before we return it.
    return float(xp.linalg.norm(error, ord=order) / xp.linalg.norm(ex, ord=order))

  @property
  def domain(self) -> Domain:
    return self._domain

  @property
  def dim(self) -> int:
    return self._dim


  @property
  def nbfaces(self) -> int:
    return self._nbfaces

  @property
  def nbcells(self) -> int:
    return self._nbcells

  @property
  def nbnodes(self) -> int:
    return self._nbnodes

  @property
  def nbhalos(self) -> int:
    return self._nbhalos

  @property
  def name(self) -> Optional[str]:
    return self._name

  @property
  def BCs(self) -> dict[str, Boundary]:
    return self._BCs

  @property
  def BCin(self) -> Boundary:
    return self._BCin

  @property
  def BCout(self) -> Boundary:
    return self._BCout

  @property
  def BCupper(self) -> Boundary:
    return self._BCupper

  @property
  def BCbottom(self) -> Boundary:
    return self._BCbottom

  @property
  def BCback(self) -> Optional[Boundary]:
    return self._BCback

  @property
  def BCfront(self) -> Optional[Boundary]:
    return self._BCfront
