#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bounded VOF phase-fraction transport (interFoam phase 1).

`VOFAdvection` transports a phase fraction alpha in [0,1] by a given divergence-free
volumetric face flux phi, with artificial interface compression to keep the interface
sharp (see vof_compute.py). The explicit update is clipped to [0,1] so alpha stays
bounded; with a divergence-free phi and closed/zero-alpha boundaries the phase volume
sum_cell alpha*V is conserved.

Stand-alone (phi prescribed) for validation now; it will be driven by the PISO
conservative face flux self._phi for the coupled two-phase solver.
"""
import numpy as np
from mpi4py import MPI

from manapy.solvers.incompressible.fvm_utils_compute import get_kernels
from manapy.solvers.incompressible.vof_compute import get_vof_kernels


def volume_fractions(domain, halfplanes):
  """Exact per-cell volume fraction of the convex region {n.x <= c for all (nx,ny,c) in
  `halfplanes`} -- successive Sutherland-Hodgman clips of each cell polygon (tri/quad).

  THE way to initialise alpha for two-phase-under-gravity: a binary 0/1 alpha carries no
  sub-cell interface position (it represents an interface jagged along the cell
  boundaries, which is NOT a discrete equilibrium), and the hydrostatic face balance is
  then wrong by O(1) at the jump, ratio-amplified into parasitic currents on triangle
  meshes. With EXACT cut-cell fractions the p_rgh face balance is exact to machine
  precision on any mesh (measured 1e-13..1e-10 at density ratio 10..1000).

  Examples: flat layer below y=b -> [(0,1,b)]; dam column x<=a, y<=b -> [(1,0,a),(0,1,b)].
  """
  nodeid = np.asarray(domain.cells.nodeid)
  vx = np.asarray(domain.nodes.vertex)
  nc = domain.nbcells
  frac = np.zeros(nc)
  for i in range(nc):
    nv = nodeid[i, -1]
    pts = [vx[nodeid[i, k], :2] for k in range(nv)]
    p = np.asarray(pts, dtype=np.float64)
    area0 = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                      np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    for (hx, hy, c) in halfplanes:
      clipped = []
      m = len(pts)
      for k in range(m):
        a = pts[k]; b = pts[(k + 1) % m]
        da = hx * a[0] + hy * a[1] - c; db = hx * b[0] + hy * b[1] - c
        if da <= 0.0:
          clipped.append(a)
        if (da < 0.0) != (db < 0.0) and da != db:
          t = da / (da - db)
          clipped.append(a + t * (b - a))
      pts = clipped
      if len(pts) < 3:
        break
    if len(pts) >= 3 and area0 > 0.0:
      p = np.asarray(pts, dtype=np.float64)
      cut = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                      np.dot(p[:, 1], np.roll(p[:, 0], -1)))
      frac[i] = min(1.0, cut / area0)
  return frac


class VOFAdvection:

  def __init__(self, alpha, cAlpha=1.0):
    """
    alpha : cell phase-fraction Variable (values in [0,1]).
    cAlpha : interface-compression coefficient (interFoam's cAlpha; 1.0 typical,
             0 disables compression -> plain bounded upwind advection).
    """
    self.alpha = alpha
    self.domain = dom = alpha.domain
    if alpha.dim != 2:
      raise NotImplementedError("VOFAdvection is wired for 2D")
    self.cAlpha = float(cAlpha)

    self.cellid = np.asarray(dom.faces.cellid, dtype=np.int64)
    self.halofid = np.asarray(dom.faces.halofid, dtype=np.int64)
    self.fname = np.asarray(dom.faces.name, dtype=np.int64)
    self.normal = np.ascontiguousarray(np.asarray(dom.faces.normal)[:, :2])
    self.vol = np.asarray(dom.cells.volume)
    self.nc = dom.nbcells
    self.nh = int(getattr(dom, "nbhalos", 0))

    # reuse the collocated Green-Gauss cell-gradient kernel for grad(alpha)
    self._face_flux, _, self._gg_grad = get_kernels()
    self._adv, self._sums, self._apply = get_vof_kernels()

    self.nf = len(self.cellid)
    self._gx = np.zeros(self.nc); self._gy = np.zeros(self.nc)
    self._gxh = np.zeros(self.nh); self._gyh = np.zeros(self.nh)
    self._res_adv = np.zeros(self.nc); self._Af = np.zeros(self.nf)
    self._Pp = np.zeros(self.nc); self._Pm = np.zeros(self.nc)
    self._res_corr = np.zeros(self.nc)
    self._Rph = np.zeros(self.nh); self._Rmh = np.zeros(self.nh)
    # bounded alpha FACE flux (low-order + limited compression) -> alphaPhi, for the
    # consistent mass flux rhoPhi in the momentum equation (Rudman consistency).
    self._aphi_lo = np.zeros(self.nf); self._aphi_hi = np.zeros(self.nf)
    self.alphaPhi = np.zeros(self.nf)

  def _grad_alpha(self):
    self.alpha.update_halo_value()
    self._gg_grad(self.alpha.cell, self.alpha.halo, self.normal, self.cellid,
                  self.halofid, self.fname, self.vol, self._gx, self._gy)
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gx), recv_buffer=self._gxh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gy), recv_buffer=self._gyh)

  def step(self, phi, dt):
    """Advance alpha one step by the volumetric face flux `phi` (len nbfaces) over dt.
    Explicit upwind advection + Zalesak (MULES)-limited interface compression:
    conservative AND bounded in [0,1] (no clipping). Returns the bounded alpha FACE flux
    `alphaPhi` (low-order + limited compression), so the caller can build the consistent
    mass flux rhoPhi = alphaPhi*(rho1-rho2) + phi*rho2 for the momentum convection. The
    alpha update itself is unchanged (bit-identical to the stand-alone brick-1 solver)."""
    self._grad_alpha()
    # 1. low-order (upwind) advection + per-face antidiffusive compression flux + the
    #    low-order alpha face flux aphi_lo
    self._adv(self.alpha.cell, self.alpha.halo, phi, self._gx, self._gy, self._gxh,
              self._gyh, self.cAlpha, self.normal, self.cellid, self.halofid,
              self.fname, self._res_adv, self._Af, self._aphi_lo)
    alpha_low = self.alpha.cell - dt * self._res_adv / self.vol
    # 2. Zalesak limiter: allowable increase/decrease per cell -> face factor lambda
    self._sums(self._Af, self.cellid, self.halofid, self.fname, self._Pp, self._Pm)
    dtv = dt / self.vol
    Rp = np.minimum(1.0, (1.0 - alpha_low) / (dtv * self._Pp + 1e-30))
    Rm = np.minimum(1.0, alpha_low / (dtv * self._Pm + 1e-30))
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(Rp), recv_buffer=self._Rph)
      self.domain.halo_comm.exchange(np.ascontiguousarray(Rm), recv_buffer=self._Rmh)
    # 3. conservative limited compression correction + its face flux aphi_hi
    self._apply(self._Af, Rp, Rm, self._Rph, self._Rmh, self.cellid, self.halofid,
                self.fname, self._res_corr, self._aphi_hi)
    a = self.alpha.cell
    a[:] = alpha_low - dt * self._res_corr / self.vol
    np.clip(a, 0.0, 1.0, out=a)                        # safety against round-off only
    self.alphaPhi[:] = self._aphi_lo + self._aphi_hi   # bounded alpha face flux (L->R)
    return self.alphaPhi

  def face_flux(self, u, v, uw=None, vw=None):
    """Helper: build a face flux phi = u_face . S_f from a cell velocity field (for the
    stand-alone prescribed-velocity tests). uw/vw are per-face boundary velocities."""
    nf = len(self.cellid)
    uw = np.zeros(nf) if uw is None else uw
    vw = np.zeros(nf) if vw is None else vw
    uh = np.zeros(self.nh); vh = np.zeros(self.nh)
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(u), recv_buffer=uh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(v), recv_buffer=vh)
    phi = np.zeros(nf)
    self._face_flux(u, v, uh, vh, uw, vw, self.normal, self.cellid, self.halofid,
                    self.fname, phi)
    return phi

  def phase_volume(self):
    """Global sum_cell alpha*V (phase-1 volume) -- conserved by a divergence-free phi."""
    loc = float(np.sum(self.alpha.cell * self.vol))
    return self.domain.comm.allreduce(loc, op=MPI.SUM)

  def bounds(self):
    """Global (min, max) of alpha -- must stay within [0,1]."""
    lo = self.domain.comm.allreduce(float(self.alpha.cell.min()), op=MPI.MIN)
    hi = self.domain.comm.allreduce(float(self.alpha.cell.max()), op=MPI.MAX)
    return lo, hi
