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


_weno_euler_compiled = None


class WenoEulerSolver:
  """WENO + Rusanov + SSP-RK3 solver for 2D compressible Euler.

  bc maps each boundary name to 'outflow' (zero-gradient) or a fixed state dict
  {'rho','u','v','p'} (supersonic inflow / Dirichlet). Conservative cell arrays
  rho, rhou, rhov, rhoE are evolved in place by `step`.
  """

  def __init__(self, domain, rho, rhou, rhov, rhoE, gamma=1.4, cfl=0.4, bc=None, weno=None):
    self.domain = domain
    self.rho = rho; self.rhou = rhou; self.rhov = rhov; self.rhoE = rhoE
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
    self.rhov_g = np.zeros(nbfaces); self.rhoE_g = np.zeros(nbfaces)
    # precompute boundary face masks per named boundary
    self._bmask = {}
    for bname, spec in self.bc.items():
      code = domain.BCs[bname][1]
      self._bmask[bname] = np.nonzero(self.face_name == code)[0]
    global _weno_euler_compiled
    if _weno_euler_compiled is None:
      _weno_euler_compiled = compile(_weno_euler_rusanov_2d)
    self._kernel = _weno_euler_compiled
    self._rez = [np.zeros(domain.nbcells) for _ in range(4)]

  def _fill_ghosts(self, rho, rhou, rhov, rhoE):
    il = self.cellid[:, 0]
    for bname, spec in self.bc.items():
      f = self._bmask[bname]
      if spec == "outflow":
        self.rho_g[f] = rho[il[f]]; self.rhou_g[f] = rhou[il[f]]
        self.rhov_g[f] = rhov[il[f]]; self.rhoE_g[f] = rhoE[il[f]]
      else:                                         # fixed state (inflow / Dirichlet)
        r = spec["rho"]; u = spec["u"]; v = spec.get("v", 0.0); p = spec["p"]
        self.rho_g[f] = r; self.rhou_g[f] = r * u; self.rhov_g[f] = r * v
        self.rhoE_g[f] = p / (self.gamma - 1.0) + 0.5 * r * (u * u + v * v)

  def residual(self, rho, rhou, rhov, rhoE):
    """d(U*vol)/dt for the four conservative variables (WENO + Rusanov)."""
    self._fill_ghosts(rho, rhou, rhov, rhoE)
    c_rho = self.W.weno_reconstruct(rho)
    c_rhou = self.W.weno_reconstruct(rhou)
    c_rhov = self.W.weno_reconstruct(rhov)
    c_rhoE = self.W.weno_reconstruct(rhoE)
    rz = self._rez
    W = self.W
    self._kernel(rz[0], rz[1], rz[2], rz[3],
                 np.ascontiguousarray(rho), np.ascontiguousarray(rhou),
                 np.ascontiguousarray(rhov), np.ascontiguousarray(rhoE),
                 c_rho, c_rhou, c_rhov, c_rhoE,
                 self.rho_g, self.rhou_g, self.rhov_g, self.rhoE_g,
                 W._ea, W._eb, W._M0_p, W._cx, W._cy, W.h, W._fcx, W._fcy,
                 self.cellid, self.normal, self.mesure, self.face_name, self.gamma)
    return rz

  def stepper(self):
    u = self.rhou_arr() / self.rho_arr()
    v = self.rhov_arr() / self.rho_arr()
    p = (self.gamma - 1.0) * (self.rhoE_arr() - 0.5 * self.rho_arr() * (u * u + v * v))
    c = np.sqrt(self.gamma * np.maximum(p, 1e-12) / self.rho_arr())
    hcell = np.sqrt(self.vol)
    dt = self.cfl * np.min(hcell / (np.sqrt(u * u + v * v) + c))
    return float(dt)

  # array accessors (work whether the fields are Variables or plain arrays)
  def rho_arr(self): return self.rho.cell if hasattr(self.rho, "cell") else self.rho
  def rhou_arr(self): return self.rhou.cell if hasattr(self.rhou, "cell") else self.rhou
  def rhov_arr(self): return self.rhov.cell if hasattr(self.rhov, "cell") else self.rhov
  def rhoE_arr(self): return self.rhoE.cell if hasattr(self.rhoE, "cell") else self.rhoE

  def step(self, dt):
    """One SSP-RK3 step (in place)."""
    r, ru, rv, rE = self.rho_arr(), self.rhou_arr(), self.rhov_arr(), self.rhoE_arr()
    vol = self.vol
    r0, ru0, rv0, rE0 = r.copy(), ru.copy(), rv.copy(), rE.copy()

    def stage(coef_old, r_o, ru_o, rv_o, rE_o):
      rz = self.residual(r, ru, rv, rE)
      r[:] = coef_old * r_o + (1 - coef_old) * (r + dt * rz[0] / vol)
      ru[:] = coef_old * ru_o + (1 - coef_old) * (ru + dt * rz[1] / vol)
      rv[:] = coef_old * rv_o + (1 - coef_old) * (rv + dt * rz[2] / vol)
      rE[:] = coef_old * rE_o + (1 - coef_old) * (rE + dt * rz[3] / vol)

    stage(0.0, r0, ru0, rv0, rE0)       # u1 = u + dt L(u)
    stage(0.75, r0, ru0, rv0, rE0)      # u2 = 3/4 u0 + 1/4 (u1 + dt L(u1))
    stage(1.0 / 3.0, r0, ru0, rv0, rE0)  # u^{n+1} = 1/3 u0 + 2/3 (u2 + dt L(u2))
