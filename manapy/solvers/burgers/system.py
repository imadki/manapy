#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit finite-volume solver for the nonlinear (viscous) Burgers equation.

    u_t + div( u^2/2 , u^2/2 )  =  nu * lap(u)

Unlike the advection-diffusion solver, the transport speed is not a prescribed
velocity field -- the solution advects itself (flux f(u) = u^2/2), so the wave
speed is |u|. The class mirrors AdvectionDiffusionSolver's interface
(stepper / compute_fluxes / compute_new_val) so it drops straight into the
high-level api time loop (manapy.api.models._ExplicitModel).
"""
from mpi4py import MPI

import manapy.solvers.burgers.fvm_utils_compute as fvm_utils_compute
from manapy.core.Variable import Variable


class BurgersSolver:
  # Numerical-flux schemes: local Lax-Friedrichs (== Rusanov for a scalar flux).
  SCHEMES = ("rusanov", "lax_friedrichs")

  def __init__(self,
               var: Variable,
               nu: float = 0.0,
               Dxx: float | None = None,
               Dyy: float | None = None,
               Dzz: float | None = None,
               dt: float = 0.0,
               order: int = 1,
               cfl: float = 0.4,
               scheme: str = "rusanov"):

    self.var = var
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.comm = self.domain.halo_comm.graph_comm

    # Isotropic viscosity `nu` unless a per-direction diffusivity is given.
    self.Dxx = nu if Dxx is None else Dxx
    self.Dyy = nu if Dyy is None else Dyy
    self.Dzz = (nu if self.dim == 3 else 0.0) if Dzz is None else Dzz

    self.dt = dt
    self.order = order
    self.cfl = cfl

    if scheme not in BurgersSolver.SCHEMES:
      raise ValueError(f"unknown scheme '{scheme}'; choose from {list(BurgersSolver.SCHEMES)}")
    self.scheme = scheme

    self.diffusion = not (self.Dxx == self.Dyy == self.Dzz == 0)

    # add_term -> backend arrays (GPUArray under GPU) so kernel writes stick.
    self.var.add_term("convective")
    self.var.add_term("dissipative")
    self.var.add_term("source")

    fvm_utils_compute.setup(self.dim, self.scheme)
    if self.domain.backend.name == "gpu":
      raise NotImplementedError("Burgers solver is implemented for the CPU backend only")
    if self.dim == 2:
      self._explicitscheme_convective = fvm_utils_compute.explicitscheme_convective_2d
    else:
      raise NotImplementedError("Burgers solver currently supports dim == 2 only")
    self._explicitscheme_dissipative = fvm_utils_compute.explicitscheme_dissipative
    self._time_step = fvm_utils_compute.time_step
    self._update_new_value = fvm_utils_compute.update_new_value

  def explicit_convective(self):
    if self.order == 2:
      self.var.compute_cell_gradient()
    self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo,
                                    self.var.gradcellx, self.var.gradcelly, self.var.gradcellz,
                                    self.var.gradhalocellx, self.var.gradhalocelly, self.var.gradhalocellz,
                                    self.var.psi, self.var.psihalo,
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
                                    self.domain.faces.cellid, self.domain.faces.normal,
                                    self.domain.faces.halofid, self.domain.faces.name,
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces,
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, self.order)

  def explicit_dissipative(self):
    self.var.compute_face_gradient()
    self._explicitscheme_dissipative(self.var.gradfacex, self.var.gradfacey, self.var.gradfacez,
                                     self.domain.faces.cellid,
                                     self.domain.faces.normal, self.domain.faces.name, self.var.dissipative,
                                     self.Dxx, self.Dyy, self.Dzz)

  def stepper(self):
    # Self-advection: the solution IS the transport velocity, so the convective
    # CFL wave speed per face is |u*(nx+ny[+nz])| -- pass var.cell as u, v, w
    # (the unused normal component carries a zero factor in 2D).
    d_t = self._time_step(self.var.cell, self.var.cell, self.var.cell, self.cfl,
                          self.domain.faces.normal, self.domain.faces.mesure,
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim,
                          self.Dxx, self.Dyy, self.Dzz)
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return self.dt

  def compute_fluxes(self):
    self.var.update_halo_value()
    self.var.update_ghost_value()

    # nonlinear convective flux
    self.explicit_convective()

    # viscous (dissipative) flux
    if self.diffusion:
      self.var.interpolate_celltonode()
      self.explicit_dissipative()

  def compute_new_val(self):
    self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source,
                           self.dt, self.domain.cells.volume)
