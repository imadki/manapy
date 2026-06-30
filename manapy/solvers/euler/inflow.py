#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulent inflow boundary condition (Phase 7).

Reproduces a synthetic-turbulence shear-layer inflow: a prescribed mean
cross-stream profile with superimposed fluctuations. The mean is a
hyperbolic-tangent shear layer between two streams,

    u_mean(y) = 1/2 (u1+u2) + 1/2 (u1-u2) tanh((y-y0)/delta),

and likewise for T and p; the turbulence is white noise scaled by the local
intensity (fluct = sqrt(2/3 k), u' = fluct*(2*rand-1) per component).
A spatially/temporally correlated variant is offered as a refinement (digital
filter) for smoother, more physical fluctuations.

The inflow conservative ghost state is written onto the chosen inflow boundary
faces. (Mixing inflow + outflow/wall on different boundaries needs the per-
boundary BC dispatch that is a separate solver refinement; this component
generates and applies the inflow state itself.)
"""
import numpy as np


class TurbulentInflow:

  def __init__(self, solver, boundary, u1, u2, p, T=None, rho1=None, rho2=None,
               delta=0.1, y0=0.0, gamma=None, R=287.0,
               intensity=0.05, correlated=False, corr_len=None, corr_time=None,
               seed=0):
    """
    solver    : EulerSolver (provides domain, faces, ghost arrays)
    boundary  : inflow boundary name (key in domain.BCs) or its integer code
    u1, u2    : the two stream velocities (high/low speed) [m/s]
    p         : inflow pressure (uniform) [Pa]
    T, rho1/2 : either give T (ideal gas -> rho from p,T) or rho1/rho2 per stream
    delta, y0 : shear-layer thickness and centre (cross-stream y)
    intensity : turbulence intensity (rms fluctuation / Delta U), or callable I(y)
    correlated: if True, use a digital-filter correlated fluctuation in time/space
    """
    self.solver = solver
    self.dim = solver.dim
    self.gamma = float(gamma if gamma is not None else solver.gamma)
    self.R = float(R)
    self.u1 = float(u1); self.u2 = float(u2)
    self.p = float(p)
    self.delta = float(delta); self.y0 = float(y0)
    self.intensity = intensity
    self.correlated = bool(correlated)
    self.rng = np.random.default_rng(seed)

    dom = solver.domain
    # resolve the boundary code
    if isinstance(boundary, str):
      code = dom.BCs[boundary][1]
    else:
      code = int(boundary)
    self.code = code
    name = np.asarray(solver.face_name)
    self.faces = np.nonzero(name == code)[0]
    fc = np.asarray(dom.faces.center)
    self.yf = fc[self.faces, 1]            # cross-stream coordinate of inflow faces
    self.nf = self.faces.size

    th = np.tanh((self.yf - self.y0) / self.delta)
    self.u_mean = 0.5 * (u1 + u2) + 0.5 * (u1 - u2) * th
    if T is not None:
      self.T = float(T) * np.ones(self.nf)
      self.rho = self.p / (self.R * self.T)
    else:
      self.rho = 0.5 * (rho1 + rho2) + 0.5 * (rho1 - rho2) * th
      self.T = self.p / (self.R * self.rho)
    dU = abs(u1 - u2) if u1 != u2 else max(abs(u1), 1.0)
    Iy = intensity(self.yf) if callable(intensity) else intensity
    self.fluct = Iy * dU * np.ones(self.nf)   # rms amplitude per component

    # state for the correlated (AR(1) in time) fluctuation
    self._prev = np.zeros((self.nf, 3))
    self.corr_time = corr_time
    self.corr_len = corr_len

  def _fluctuations(self, dt=None):
    """Return (u', v', w') of shape (nf, 3) with the target rms self.fluct."""
    if not self.correlated:
      # white noise: (2*rand-1) has variance 1/3 -> scale to unit variance
      w = (2.0 * self.rng.random((self.nf, 3)) - 1.0) * np.sqrt(3.0)
      return w * self.fluct[:, None]
    # AR(1) temporal correlation: x_{n+1} = a x_n + sqrt(1-a^2) noise
    a = 0.0 if (self.corr_time is None or dt is None) else np.exp(-dt / self.corr_time)
    noise = self.rng.standard_normal((self.nf, 3))
    if self.corr_len is not None:
      noise = self._spatial_filter(noise)
    self._prev = a * self._prev + np.sqrt(max(1.0 - a * a, 1e-12)) * noise
    return self._prev * self.fluct[:, None]

  def _spatial_filter(self, noise):
    """Crude 1-D Gaussian smoothing of the noise along the (sorted) face line."""
    order = np.argsort(self.yf)
    y = self.yf[order]
    out = np.empty_like(noise)
    n = self.nf
    for i in range(n):
      w = np.exp(-0.5 * ((y - y[i]) / self.corr_len) ** 2)
      w /= w.sum()
      out[order[i]] = (w[:, None] * noise[order]).sum(axis=0)
    # renormalise to unit variance after smoothing
    for c in range(noise.shape[1]):
      s = out[:, c].std()
      if s > 0:
        out[:, c] /= s
    return out

  def state(self, dt=None):
    """Return inflow conservative state arrays for the inflow faces."""
    up = self._fluctuations(dt)
    u = self.u_mean + up[:, 0]
    v = up[:, 1]
    w = up[:, 2]
    rho = self.rho
    rhou = rho * u
    rhov = rho * v
    rhow = rho * w
    ke = 0.5 * rho * (u * u + v * v + (w * w if self.dim == 3 else 0.0))
    rhoE = self.p / (self.gamma - 1.0) + ke
    return rho, rhou, rhov, rhow, rhoE

  def apply(self, dt=None):
    """Write the inflow conservative state onto the inflow-boundary ghost arrays."""
    s = self.solver
    rho, rhou, rhov, rhow, rhoE = self.state(dt)
    f = self.faces
    s.rho.ghost[f] = rho
    s.P.ghost[f] = self.p
    s.rhou.ghost[f] = rhou
    s.rhov.ghost[f] = rhov
    s.rhoE.ghost[f] = rhoE
    s.ug[f] = rhou / rho
    s.vg[f] = rhov / rho
    if self.dim == 3:
      s.rhow.ghost[f] = rhow
      s.wg[f] = rhow / rho
