#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WENO finite-volume solver for the 2D compressible Euler system.

Couples the unstructured WENO reconstruction (`weno.py`, Tsoutsanis JCP 2023) to a
Rusanov Riemann flux: each conservative variable (rho, rho*u, rho*v, rho*E) is
WENO-reconstructed, the left/right states are evaluated at the face centres, and
the Rusanov flux is taken between them. Time integration is SSP-RK3 (the standard
high-order, stable pairing for WENO). Boundary states are supplied as ghost arrays
(filled per boundary by the caller / `WenoEulerSolver`).

The per-step flux is a single compiled (numba) kernel over the precomputed
WENO/mesh data.
"""
import numpy as np
from manapy.backends.compile_fun import compile

from manapy.solvers.euler.weno import WenoReconstruction


def _weno_euler_rusanov_2d(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhoE: 'float64[:]',
                           rho: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', rhoE: 'float64[:]',
                           c_rho: 'float64[:,:]', c_rhou: 'float64[:,:]', c_rhov: 'float64[:,:]', c_rhoE: 'float64[:,:]',
                           rho_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhoE_g: 'float64[:]',
                           ea: 'int32[:]', eb: 'int32[:]', M0: 'float64[:,:]',
                           cx: 'float64[:]', cy: 'float64[:]', h: 'float64[:]', fcx: 'float64[:]', fcy: 'float64[:]',
                           cellid: 'int32[:,:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                           name: 'uint32[:]', gamma: 'float64'):
    K = c_rho.shape[1]
    rez_rho[:] = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))
    nbface = len(cellid)
    for f in range(nbface):
        mes = mesure[f]
        nx = normal[f][0] / mes
        ny = normal[f][1] / mes
        il = cellid[f][0]
        # --- left (owner) WENO state at the face centre ---
        hxl = (fcx[f] - cx[il]) / h[il]
        hyl = (fcy[f] - cy[il]) / h[il]
        rL = rho[il]; ruL = rhou[il]; rvL = rhov[il]; rEL = rhoE[il]
        for k in range(K):
            phi = hxl ** ea[k] * hyl ** eb[k] - M0[il, k]
            rL += c_rho[il, k] * phi
            ruL += c_rhou[il, k] * phi
            rvL += c_rhov[il, k] * phi
            rEL += c_rhoE[il, k] * phi
        # --- right state ---
        if name[f] == 0:
            ir = cellid[f][1]
            hxr = (fcx[f] - cx[ir]) / h[ir]
            hyr = (fcy[f] - cy[ir]) / h[ir]
            rR = rho[ir]; ruR = rhou[ir]; rvR = rhov[ir]; rER = rhoE[ir]
            for k in range(K):
                phi = hxr ** ea[k] * hyr ** eb[k] - M0[ir, k]
                rR += c_rho[ir, k] * phi
                ruR += c_rhou[ir, k] * phi
                rvR += c_rhov[ir, k] * phi
                rER += c_rhoE[ir, k] * phi
        else:                                       # boundary: ghost (first order)
            rR = rho_g[f]; ruR = rhou_g[f]; rvR = rhov_g[f]; rER = rhoE_g[f]

        uL = ruL / rL; vL = rvL / rL
        uR = ruR / rR; vR = rvR / rR
        pL = (gamma - 1.0) * (rEL - 0.5 * (ruL * ruL + rvL * rvL) / rL)
        pR = (gamma - 1.0) * (rER - 0.5 * (ruR * ruR + rvR * rvR) / rR)
        cl = np.sqrt(gamma * pL / rL) if pL > 0 else 0.0
        cr = np.sqrt(gamma * pR / rR) if pR > 0 else 0.0
        unL = uL * nx + vL * ny
        unR = uR * nx + vR * ny
        sL = np.fabs(unL) + cl
        sR = np.fabs(unR) + cr
        S = sL if sL > sR else sR

        fl_rho = rL * unL
        fl_rhou = ruL * unL + pL * nx
        fl_rhov = rvL * unL + pL * ny
        fl_rhoE = (rEL + pL) * unL
        fr_rho = rR * unR
        fr_rhou = ruR * unR + pR * nx
        fr_rhov = rvR * unR + pR * ny
        fr_rhoE = (rER + pR) * unR

        f_rho = (0.5 * (fl_rho + fr_rho) - 0.5 * S * (rR - rL)) * mes
        f_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5 * S * (ruR - ruL)) * mes
        f_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5 * S * (rvR - rvL)) * mes
        f_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5 * S * (rER - rEL)) * mes

        rez_rho[il] -= f_rho
        rez_rhou[il] -= f_rhou
        rez_rhov[il] -= f_rhov
        rez_rhoE[il] -= f_rhoE
        if name[f] == 0:
            ir = cellid[f][1]
            rez_rho[ir] += f_rho
            rez_rhou[ir] += f_rhou
            rez_rhov[ir] += f_rhov
            rez_rhoE[ir] += f_rhoE


def _weno_euler_rusanov_3d(rez_rho: 'float64[:]', rez_rhou: 'float64[:]', rez_rhov: 'float64[:]', rez_rhow: 'float64[:]', rez_rhoE: 'float64[:]',
                           rho: 'float64[:]', rhou: 'float64[:]', rhov: 'float64[:]', rhow: 'float64[:]', rhoE: 'float64[:]',
                           c_rho: 'float64[:,:]', c_rhou: 'float64[:,:]', c_rhov: 'float64[:,:]', c_rhow: 'float64[:,:]', c_rhoE: 'float64[:,:]',
                           rho_g: 'float64[:]', rhou_g: 'float64[:]', rhov_g: 'float64[:]', rhow_g: 'float64[:]', rhoE_g: 'float64[:]',
                           ea: 'int32[:]', eb: 'int32[:]', ec: 'int32[:]', M0: 'float64[:,:]',
                           cx: 'float64[:]', cy: 'float64[:]', cz: 'float64[:]', h: 'float64[:]',
                           fcx: 'float64[:]', fcy: 'float64[:]', fcz: 'float64[:]',
                           cellid: 'int32[:,:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                           name: 'uint32[:]', gamma: 'float64'):
    K = c_rho.shape[1]
    rez_rho[:] = np.zeros(len(rez_rho))
    rez_rhou[:] = np.zeros(len(rez_rhou))
    rez_rhov[:] = np.zeros(len(rez_rhov))
    rez_rhow[:] = np.zeros(len(rez_rhow))
    rez_rhoE[:] = np.zeros(len(rez_rhoE))
    nbface = len(cellid)
    for f in range(nbface):
        mes = mesure[f]
        nx = normal[f][0] / mes
        ny = normal[f][1] / mes
        nz = normal[f][2] / mes
        il = cellid[f][0]
        hxl = (fcx[f] - cx[il]) / h[il]
        hyl = (fcy[f] - cy[il]) / h[il]
        hzl = (fcz[f] - cz[il]) / h[il]
        rL = rho[il]; ruL = rhou[il]; rvL = rhov[il]; rwL = rhow[il]; rEL = rhoE[il]
        for k in range(K):
            phi = hxl ** ea[k] * hyl ** eb[k] * hzl ** ec[k] - M0[il, k]
            rL += c_rho[il, k] * phi
            ruL += c_rhou[il, k] * phi
            rvL += c_rhov[il, k] * phi
            rwL += c_rhow[il, k] * phi
            rEL += c_rhoE[il, k] * phi
        if name[f] == 0:
            ir = cellid[f][1]
            hxr = (fcx[f] - cx[ir]) / h[ir]
            hyr = (fcy[f] - cy[ir]) / h[ir]
            hzr = (fcz[f] - cz[ir]) / h[ir]
            rR = rho[ir]; ruR = rhou[ir]; rvR = rhov[ir]; rwR = rhow[ir]; rER = rhoE[ir]
            for k in range(K):
                phi = hxr ** ea[k] * hyr ** eb[k] * hzr ** ec[k] - M0[ir, k]
                rR += c_rho[ir, k] * phi
                ruR += c_rhou[ir, k] * phi
                rvR += c_rhov[ir, k] * phi
                rwR += c_rhow[ir, k] * phi
                rER += c_rhoE[ir, k] * phi
        else:
            rR = rho_g[f]; ruR = rhou_g[f]; rvR = rhov_g[f]; rwR = rhow_g[f]; rER = rhoE_g[f]

        uL = ruL / rL; vL = rvL / rL; wL = rwL / rL
        uR = ruR / rR; vR = rvR / rR; wR = rwR / rR
        pL = (gamma - 1.0) * (rEL - 0.5 * (ruL * ruL + rvL * rvL + rwL * rwL) / rL)
        pR = (gamma - 1.0) * (rER - 0.5 * (ruR * ruR + rvR * rvR + rwR * rwR) / rR)
        cl = np.sqrt(gamma * pL / rL) if pL > 0 else 0.0
        cr = np.sqrt(gamma * pR / rR) if pR > 0 else 0.0
        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz
        sL = np.fabs(unL) + cl
        sR = np.fabs(unR) + cr
        S = sL if sL > sR else sR

        fl_rho = rL * unL
        fl_rhou = ruL * unL + pL * nx
        fl_rhov = rvL * unL + pL * ny
        fl_rhow = rwL * unL + pL * nz
        fl_rhoE = (rEL + pL) * unL
        fr_rho = rR * unR
        fr_rhou = ruR * unR + pR * nx
        fr_rhov = rvR * unR + pR * ny
        fr_rhow = rwR * unR + pR * nz
        fr_rhoE = (rER + pR) * unR

        f_rho = (0.5 * (fl_rho + fr_rho) - 0.5 * S * (rR - rL)) * mes
        f_rhou = (0.5 * (fl_rhou + fr_rhou) - 0.5 * S * (ruR - ruL)) * mes
        f_rhov = (0.5 * (fl_rhov + fr_rhov) - 0.5 * S * (rvR - rvL)) * mes
        f_rhow = (0.5 * (fl_rhow + fr_rhow) - 0.5 * S * (rwR - rwL)) * mes
        f_rhoE = (0.5 * (fl_rhoE + fr_rhoE) - 0.5 * S * (rER - rEL)) * mes

        rez_rho[il] -= f_rho
        rez_rhou[il] -= f_rhou
        rez_rhov[il] -= f_rhov
        rez_rhow[il] -= f_rhow
        rez_rhoE[il] -= f_rhoE
        if name[f] == 0:
            ir = cellid[f][1]
            rez_rho[ir] += f_rho
            rez_rhou[ir] += f_rhou
            rez_rhov[ir] += f_rhov
            rez_rhow[ir] += f_rhow
            rez_rhoE[ir] += f_rhoE


_weno_euler_compiled = None
_weno_euler_3d_compiled = None


class WenoEulerSolver:
  """WENO + Rusanov + SSP-RK3 solver for 2D compressible Euler.

  bc maps each boundary name to 'outflow' (zero-gradient) or a fixed state dict
  {'rho','u','v','p'} (supersonic inflow / Dirichlet). Conservative cell arrays
  rho, rhou, rhov, rhoE are evolved in place by `step`.
  """

  def __init__(self, domain, rho, rhou, rhov, rhoE, rhow=None, gamma=1.4, cfl=0.4, bc=None, weno=None):
    self.domain = domain
    self.dim = int(getattr(domain, "dim", 2))
    self.rho = rho; self.rhou = rhou; self.rhov = rhov; self.rhoE = rhoE; self.rhow = rhow
    if self.dim == 3 and rhow is None:
      raise ValueError("3D WenoEulerSolver requires the rhow field")
    self.gamma = float(gamma)
    self.cfl = float(cfl)
    self.W = weno if weno is not None else WenoReconstruction(domain, order=2)
    self.bc = bc or {}
    nbfaces = np.asarray(domain.faces.name).shape[0]
    self.face_name = np.asarray(domain.faces.name, dtype=np.uint32)
    self.cellid = np.asarray(domain.faces.cellid, dtype=np.int32)
    self.normal = np.asarray(domain.faces.normal)
    self.mesure = np.asarray(domain.faces.mesure)
    self.vol = np.asarray(domain.cells.volume)
    self.rho_g = np.zeros(nbfaces); self.rhou_g = np.zeros(nbfaces)
    self.rhov_g = np.zeros(nbfaces); self.rhow_g = np.zeros(nbfaces); self.rhoE_g = np.zeros(nbfaces)
    # precompute boundary face masks per named boundary
    self._bmask = {}
    for bname, spec in self.bc.items():
      code = domain.BCs[bname][1]
      self._bmask[bname] = np.nonzero(self.face_name == code)[0]
    global _weno_euler_compiled, _weno_euler_3d_compiled
    if self.dim == 2:
      if _weno_euler_compiled is None:
        _weno_euler_compiled = compile(_weno_euler_rusanov_2d)
      self._kernel = _weno_euler_compiled
      self._rez = [np.zeros(domain.nbcells) for _ in range(4)]
    else:
      if _weno_euler_3d_compiled is None:
        _weno_euler_3d_compiled = compile(_weno_euler_rusanov_3d)
      self._kernel = _weno_euler_3d_compiled
      self._rez = [np.zeros(domain.nbcells) for _ in range(5)]

  def _fill_ghosts(self):
    il = self.cellid[:, 0]
    rho, rhou, rhov, rhoE = self.rho_arr(), self.rhou_arr(), self.rhov_arr(), self.rhoE_arr()
    rhow = self.rhow_arr()
    for bname, spec in self.bc.items():
      f = self._bmask[bname]
      if spec == "outflow":
        self.rho_g[f] = rho[il[f]]; self.rhou_g[f] = rhou[il[f]]
        self.rhov_g[f] = rhov[il[f]]; self.rhoE_g[f] = rhoE[il[f]]
        if self.dim == 3:
          self.rhow_g[f] = rhow[il[f]]
      else:                                         # fixed state (inflow / Dirichlet)
        r = spec["rho"]; u = spec["u"]; v = spec.get("v", 0.0); w = spec.get("w", 0.0); p = spec["p"]
        self.rho_g[f] = r; self.rhou_g[f] = r * u; self.rhov_g[f] = r * v; self.rhow_g[f] = r * w
        self.rhoE_g[f] = p / (self.gamma - 1.0) + 0.5 * r * (u * u + v * v + w * w)

  def residual(self):
    """d(U*vol)/dt for the conservative variables (WENO + Rusanov)."""
    self._fill_ghosts()
    W = self.W
    rz = self._rez
    rho, rhou, rhov, rhoE = self.rho_arr(), self.rhou_arr(), self.rhov_arr(), self.rhoE_arr()
    c_rho = W.weno_reconstruct(rho); c_rhou = W.weno_reconstruct(rhou)
    c_rhov = W.weno_reconstruct(rhov); c_rhoE = W.weno_reconstruct(rhoE)
    if self.dim == 2:
      self._kernel(rz[0], rz[1], rz[2], rz[3],
                   np.ascontiguousarray(rho), np.ascontiguousarray(rhou),
                   np.ascontiguousarray(rhov), np.ascontiguousarray(rhoE),
                   c_rho, c_rhou, c_rhov, c_rhoE,
                   self.rho_g, self.rhou_g, self.rhov_g, self.rhoE_g,
                   W._ea, W._eb, W._M0_p, W._cx, W._cy, W.h, W._fcx, W._fcy,
                   self.cellid, self.normal, self.mesure, self.face_name, self.gamma)
    else:
      rhow = self.rhow_arr()
      c_rhow = W.weno_reconstruct(rhow)
      self._kernel(rz[0], rz[1], rz[2], rz[3], rz[4],
                   np.ascontiguousarray(rho), np.ascontiguousarray(rhou),
                   np.ascontiguousarray(rhov), np.ascontiguousarray(rhow), np.ascontiguousarray(rhoE),
                   c_rho, c_rhou, c_rhov, c_rhow, c_rhoE,
                   self.rho_g, self.rhou_g, self.rhov_g, self.rhow_g, self.rhoE_g,
                   W._ea, W._eb, W._ec, W._M0_p, W._cx, W._cy, W._cz, W.h, W._fcx, W._fcy, W._fcz,
                   self.cellid, self.normal, self.mesure, self.face_name, self.gamma)
    return rz

  def stepper(self):
    r = self.rho_arr()
    u = self.rhou_arr() / r; v = self.rhov_arr() / r
    q2 = u * u + v * v
    if self.dim == 3:
      w = self.rhow_arr() / r; q2 = q2 + w * w
    p = (self.gamma - 1.0) * (self.rhoE_arr() - 0.5 * r * q2)
    c = np.sqrt(self.gamma * np.maximum(p, 1e-12) / r)
    hcell = self.vol ** (1.0 / self.dim)
    dt = self.cfl * np.min(hcell / (np.sqrt(q2) + c))
    return float(dt)

  # array accessors (work whether the fields are Variables or plain arrays)
  def rho_arr(self): return self.rho.cell if hasattr(self.rho, "cell") else self.rho
  def rhou_arr(self): return self.rhou.cell if hasattr(self.rhou, "cell") else self.rhou
  def rhov_arr(self): return self.rhov.cell if hasattr(self.rhov, "cell") else self.rhov
  def rhoE_arr(self): return self.rhoE.cell if hasattr(self.rhoE, "cell") else self.rhoE
  def rhow_arr(self):
    if self.rhow is None:
      return None
    return self.rhow.cell if hasattr(self.rhow, "cell") else self.rhow

  def step(self, dt):
    """One SSP-RK3 step (in place)."""
    fields = [self.rho_arr(), self.rhou_arr(), self.rhov_arr(), self.rhoE_arr()]
    if self.dim == 3:
      fields.insert(3, self.rhow_arr())             # order: rho, rhou, rhov, rhow, rhoE
    vol = self.vol
    old = [a.copy() for a in fields]

    def stage(coef_old):
      rz = self.residual()
      for a, a0, r in zip(fields, old, rz):
        a[:] = coef_old * a0 + (1 - coef_old) * (a + dt * r / vol)

    stage(0.0)          # u1 = u + dt L(u)
    stage(0.75)         # u2 = 3/4 u0 + 1/4 (u1 + dt L(u1))
    stage(1.0 / 3.0)    # u^{n+1} = 1/3 u0 + 2/3 (u2 + dt L(u2))
