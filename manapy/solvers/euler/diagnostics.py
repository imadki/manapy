#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulence diagnostics for the compressible (Euler/NS/LES) solver.

Cell-centred, instantaneous fields derived from the resolved velocity gradients
-- unstructured-FV vorticity / enstrophy / budget
post-processing (module_budget_vorticity.f90, module_tools_post.f90):

  * vorticity        omega = curl(u)         (scalar in 2D, vector in 3D)
  * enstrophy        0.5 |omega|^2
  * Q-criterion      0.5 ( |Omega|^2 - |S|^2 )   (vortex-core indicator)
  * resolved TKE     0.5 <rho |u - <u>|^2> / <rho>   (volume-weighted scalar)

Works for any run (inviscid included): it builds its own velocity Variables to
get gradients, so it does not depend on the viscous/LES path being active.

Usage:
    diag = FlowDiagnostics(rho, rhou, rhov, rhoE, rhow=rhow)   # 3D: pass rhow
    diag.compute()
    diag.vorticityz   # 2D scalar field, or diag.vorticityx/y/z + diag.vortmag (3D)
    diag.enstrophy
    diag.qcriterion
    ke_turb = diag.tke()
"""
import numpy as np
from mpi4py import MPI

from manapy.core.Variable import Variable


class FlowDiagnostics:

  def __init__(self, rho: Variable, rhou: Variable, rhov: Variable, rhoE: Variable,
               rhow: Variable = None):
    self.rho = rho
    self.rhou = rhou
    self.rhov = rhov
    self.rhow = rhow
    self.rhoE = rhoE
    self.domain = rho.domain
    self.dim = rho.dim
    self.comm = self.domain.halo_comm.graph_comm
    nbcells = self.domain.nbcells
    dtype = np.asarray(rho.cell).dtype

    # velocity Variables, only used to obtain cell gradients
    self._u = Variable(domain=self.domain)
    self._v = Variable(domain=self.domain)
    self._w = Variable(domain=self.domain) if self.dim == 3 else None

    # output fields
    self.enstrophy = np.zeros(nbcells, dtype=dtype)
    self.qcriterion = np.zeros(nbcells, dtype=dtype)
    if self.dim == 2:
      self.vorticityz = np.zeros(nbcells, dtype=dtype)
    else:
      self.vorticityx = np.zeros(nbcells, dtype=dtype)
      self.vorticityy = np.zeros(nbcells, dtype=dtype)
      self.vorticityz = np.zeros(nbcells, dtype=dtype)
      self.vortmag = np.zeros(nbcells, dtype=dtype)

  def _fill_velocity_gradients(self):
    rho = self.rho.cell
    self._u.cell[:] = self.rhou.cell / rho
    self._v.cell[:] = self.rhov.cell / rho
    vel = [self._u, self._v]
    if self.dim == 3:
      self._w.cell[:] = self.rhow.cell / rho
      vel.append(self._w)
    for var in vel:
      var.update_halo_value()
      var.update_ghost_value()
      var.compute_cell_gradient()

  def compute(self):
    """Fill vorticity, enstrophy and the Q-criterion from the current state."""
    self._fill_velocity_gradients()
    ux, uy = self._u.gradcellx, self._u.gradcelly
    vx, vy = self._v.gradcellx, self._v.gradcelly

    if self.dim == 2:
      wz = vx - uy                      # only non-zero vorticity component
      self.vorticityz[:] = wz
      self.enstrophy[:] = 0.5 * wz * wz
      # strain S and rotation Omega rate magnitudes
      s11 = ux; s22 = vy; s12 = 0.5 * (uy + vx)
      ss = s11 * s11 + s22 * s22 + 2.0 * s12 * s12
      om2 = 0.5 * (uy - vx) ** 2            # |Omega|^2 = 2*Omega12^2, Omega12=0.5(uy-vx)
      self.qcriterion[:] = 0.5 * (om2 - ss)
    else:
      uz = self._u.gradcellz
      vz = self._v.gradcellz
      wx, wy, wzz = self._w.gradcellx, self._w.gradcelly, self._w.gradcellz
      ox = wy - vz
      oy = uz - wx
      oz = vx - uy
      self.vorticityx[:] = ox
      self.vorticityy[:] = oy
      self.vorticityz[:] = oz
      self.vortmag[:] = np.sqrt(ox * ox + oy * oy + oz * oz)
      self.enstrophy[:] = 0.5 * (ox * ox + oy * oy + oz * oz)
      # |S|^2 and |Omega|^2 from the symmetric / antisymmetric gradient parts
      s11 = ux; s22 = vy; s33 = wzz
      s12 = 0.5 * (uy + vx); s13 = 0.5 * (uz + wx); s23 = 0.5 * (vz + wy)
      ss = s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
      o12 = 0.5 * (uy - vx); o13 = 0.5 * (uz - wx); o23 = 0.5 * (vz - wy)
      oo = 2.0 * (o12 * o12 + o13 * o13 + o23 * o23)
      self.qcriterion[:] = 0.5 * (oo - ss)

  def tke(self):
    """Volume-weighted resolved turbulent kinetic energy about the mean velocity.

    Mean velocity is the (mass-weighted) volume average; returns the global
    0.5 <rho |u'|^2> / <rho>.
    """
    vol = self.domain.cells.volume[:]
    rho = self.rho.cell[:]
    u = self.rhou.cell[:] / rho
    v = self.rhov.cell[:] / rho
    mrho = self.comm.allreduce(float(np.sum(vol * rho)), op=MPI.SUM)
    um = self.comm.allreduce(float(np.sum(vol * rho * u)), op=MPI.SUM) / mrho
    vm = self.comm.allreduce(float(np.sum(vol * rho * v)), op=MPI.SUM) / mrho
    up = u - um
    vp = v - vm
    sq = up * up + vp * vp
    if self.dim == 3:
      w = self.rhow.cell[:] / rho
      wm = self.comm.allreduce(float(np.sum(vol * rho * w)), op=MPI.SUM) / mrho
      wp = w - wm
      sq = sq + wp * wp
    num = self.comm.allreduce(float(np.sum(vol * 0.5 * rho * sq)), op=MPI.SUM)
    den = self.comm.allreduce(float(np.sum(vol)), op=MPI.SUM)
    return num / den
