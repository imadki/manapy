#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOF (Volume-Of-Fluid) kernels for the two-phase interFoam-style solver -- phase 1:
bounded, CONSERVATIVE transport of the phase fraction alpha with interface compression.

The phase fraction alpha in [0,1] marks fluid 1 (alpha=1) vs fluid 2 (alpha=0); the
interface is the 0<alpha<1 band. It is transported by the (divergence-free) volumetric
face flux phi, sharpened by an artificial compression term (OpenFOAM's interFoam):

    d(alpha)/dt + div(alpha u) + div(alpha(1-alpha) u_r) = 0,   u_r = cAlpha |u| n_hat

The advective part is first-order upwind (a monotone, bounded LOW-order scheme). The
compression term is an ANTIDIFFUSIVE flux that sharpens the interface but would push
alpha out of [0,1]; instead of clipping (which loses/creates mass) we bound it with a
Zalesak (MULES) flux limiter: each antidiffusive face flux is scaled by lambda_f in
[0,1] so the result stays in [0,1], with the SAME lambda_f on both sides of a face ->
conservative. This is the essence of interFoam's MULES.

Three stages (with intermediate cell/face arrays):
  1. `_alpha_adv_antidiff` : low-order advective residual + per-face antidiffusive flux Af
  2. `_zalesak_sums`       : per-cell antidiffusive in/out sums (-> allowable factors R+/R-)
  3. `_zalesak_apply`      : per-face limiter lambda_f -> conservative bounded correction

Faces: name 0 interior, 10 partition (MPI halo neighbour), else physical boundary.
"""
import numpy as np
from manapy.backends.compile_fun import compile


def _alpha_adv_antidiff_2d(alpha: 'float[:]', alpha_h: 'float[:]', phi: 'float[:]',
                           gax: 'float[:]', gay: 'float[:]', gax_h: 'float[:]', gay_h: 'float[:]',
                           calpha: 'float', normal: 'float[:,:]', cellid: 'int64[:,:]',
                           halofid: 'int64[:]', fname: 'int64[:]', res_adv: 'float[:]',
                           Af: 'float[:]', aphi_lo: 'float[:]'):
  # res_adv_P = sum_f phi_f alpha_up (upwind, monotone). Af[f] = compression flux
  # phir_f * (alpha(1-alpha))_up (antidiffusive, oriented owner->neighbour).
  # aphi_lo[f] = phi_f alpha_up = the low-order alpha FACE flux (L->R), emitted so the
  # solver can build the consistent mass flux rhoPhi (= alphaPhi*(rho1-rho2)+phi*rho2)
  # for the momentum convection (Rudman consistency). res_adv is unchanged (bit-identical
  # alpha update): res_adv == div(aphi_lo)*vol by construction.
  n = len(res_adv); nfc = len(cellid)
  for i in range(n):
    res_adv[i] = 0.0
  for f in range(nfc):
    iL = cellid[f, 0]; ph = phi[f]; aL = alpha[iL]
    nx = normal[f, 0]; ny = normal[f, 1]
    area = np.sqrt(nx * nx + ny * ny)
    if fname[f] == 0:
      iR = cellid[f, 1]; aR = alpha[iR]
      gx = 0.5 * (gax[iL] + gax[iR]); gy = 0.5 * (gay[iL] + gay[iR])
      gmag = np.sqrt(gx * gx + gy * gy) + 1e-30
      phir = calpha * (ph if ph > 0.0 else -ph) * (gx * nx + gy * ny) / (gmag * area)
      a_up = aL if ph > 0.0 else aR
      ac_up = aL * (1.0 - aL) if phir > 0.0 else aR * (1.0 - aR)
      fadv = ph * a_up
      res_adv[iL] += fadv; res_adv[iR] -= fadv
      Af[f] = phir * ac_up
      aphi_lo[f] = fadv
    elif fname[f] == 10:
      h = halofid[f]; aR = alpha_h[h]
      gx = 0.5 * (gax[iL] + gax_h[h]); gy = 0.5 * (gay[iL] + gay_h[h])
      gmag = np.sqrt(gx * gx + gy * gy) + 1e-30
      phir = calpha * (ph if ph > 0.0 else -ph) * (gx * nx + gy * ny) / (gmag * area)
      a_up = aL if ph > 0.0 else aR
      ac_up = aL * (1.0 - aL) if phir > 0.0 else aR * (1.0 - aR)
      res_adv[iL] += ph * a_up
      Af[f] = phir * ac_up
      aphi_lo[f] = ph * a_up
    else:                                              # physical boundary: outflow-only
      res_adv[iL] += ph * aL
      Af[f] = 0.0
      aphi_lo[f] = ph * aL


def _zalesak_sums_2d(Af: 'float[:]', cellid: 'int64[:,:]', halofid: 'int64[:]',
                     fname: 'int64[:]', Pp: 'float[:]', Pm: 'float[:]'):
  # Per cell: Pp = total antidiffusive flux that would INCREASE alpha, Pm = that which
  # would DECREASE it. Af>0 leaves the owner (owner decreases, neighbour increases).
  n = len(Pp); nfc = len(cellid)
  for i in range(n):
    Pp[i] = 0.0; Pm[i] = 0.0
  for f in range(nfc):
    A = Af[f]; iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      if A > 0.0:
        Pm[iL] += A; Pp[iR] += A
      else:
        Pp[iL] += -A; Pm[iR] += -A
    elif fname[f] == 10:
      if A > 0.0:
        Pm[iL] += A
      else:
        Pp[iL] += -A


def _zalesak_apply_2d(Af: 'float[:]', Rp: 'float[:]', Rm: 'float[:]', Rp_h: 'float[:]',
                      Rm_h: 'float[:]', cellid: 'int64[:,:]', halofid: 'int64[:]',
                      fname: 'int64[:]', res_corr: 'float[:]', aphi_hi: 'float[:]'):
  # Limited antidiffusive residual: lambda_f = min(donor R-, receiver R+); the SAME
  # lambda scales the flux on both sides (conservative). aphi_hi[f] = lambda_f Af[f] is
  # the limited compression FACE flux (L->R), emitted so alphaPhi = aphi_lo + aphi_hi is
  # the full bounded alpha face flux for the consistent mass flux rhoPhi. res_corr is
  # unchanged (bit-identical alpha update): res_corr == div(aphi_hi)*vol.
  n = len(res_corr); nfc = len(cellid)
  for i in range(n):
    res_corr[i] = 0.0
  for f in range(nfc):
    aphi_hi[f] = 0.0                                    # boundary faces (Af=0) stay 0
  for f in range(nfc):
    A = Af[f]; iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      if A > 0.0:
        lam = Rm[iL] if Rm[iL] < Rp[iR] else Rp[iR]
      else:
        lam = Rp[iL] if Rp[iL] < Rm[iR] else Rm[iR]
      c = lam * A
      res_corr[iL] += c; res_corr[iR] -= c
      aphi_hi[f] = c
    elif fname[f] == 10:
      h = halofid[f]
      if A > 0.0:
        lam = Rm[iL] if Rm[iL] < Rp_h[h] else Rp_h[h]
      else:
        lam = Rp[iL] if Rp[iL] < Rm_h[h] else Rm_h[h]
      res_corr[iL] += lam * A
      aphi_hi[f] = lam * A


def _gg_div_2d(cx: 'float[:]', cy: 'float[:]', cx_h: 'float[:]', cy_h: 'float[:]',
               normal: 'float[:,:]', cellid: 'int64[:,:]', halofid: 'int64[:]',
               fname: 'int64[:]', vol: 'float[:]', div: 'float[:]'):
  # Green-Gauss cell divergence of a vector field (cx, cy) -- used for the interface
  # curvature kappa = -div(n_hat).
  n = len(vol); nfc = len(cellid)
  for i in range(n):
    div[i] = 0.0
  for f in range(nfc):
    iL = cellid[f, 0]; nx = normal[f, 0]; ny = normal[f, 1]
    if fname[f] == 0:
      iR = cellid[f, 1]
      flux = 0.5 * (cx[iL] + cx[iR]) * nx + 0.5 * (cy[iL] + cy[iR]) * ny
      div[iL] += flux; div[iR] -= flux
    elif fname[f] == 10:
      h = halofid[f]
      div[iL] += 0.5 * (cx[iL] + cx_h[h]) * nx + 0.5 * (cy[iL] + cy_h[h]) * ny
    else:
      div[iL] += cx[iL] * nx + cy[iL] * ny
  for i in range(n):
    div[i] /= vol[i]


def _st_face_flux_2d(alpha: 'float[:]', alpha_h: 'float[:]', kappa: 'float[:]',
                     kappa_h: 'float[:]', sigma: 'float', Df: 'float[:]',
                     cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                     phist: 'float[:]'):
  # Well-balanced surface-tension face flux phist_f = D_f sigma K_f (alpha_N - alpha_P),
  # K_f the face-averaged curvature. Added to phiHbyA so the pressure solve balances it
  # (zero spurious flux + correct Laplace jump at equilibrium). No flux at boundaries.
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      Kf = 0.5 * (kappa[iL] + kappa[iR])
      phist[f] = Df[f] * sigma * Kf * (alpha[iR] - alpha[iL])
    elif fname[f] == 10:
      h = halofid[f]
      Kf = 0.5 * (kappa[iL] + kappa_h[h])
      phist[f] = Df[f] * sigma * Kf * (alpha_h[h] - alpha[iL])
    else:
      phist[f] = 0.0


_compiled = {}


def get_vof_kernels():
  if not _compiled:
    _compiled['adv'] = compile(_alpha_adv_antidiff_2d)
    _compiled['sums'] = compile(_zalesak_sums_2d)
    _compiled['apply'] = compile(_zalesak_apply_2d)
  return _compiled['adv'], _compiled['sums'], _compiled['apply']


def _buoy_face_flux_2d(rho: 'float[:]', rho_h: 'float[:]', ghf: 'float[:]', Df: 'float[:]',
                       cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                       phi: 'float[:]'):
  # p_rgh buoyancy face flux, ADDED to phi: phibuoy_f = -D_f (g.x)_f (rho_N - rho_P).
  # This is the interFoam `phig`; with p_rgh it makes the hydrostatic balance exact
  # (no spurious currents from the density jump under gravity). No flux at boundaries.
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      phi[f] += -Df[f] * ghf[f] * (rho[cellid[f, 1]] - rho[iL])
    elif fname[f] == 10:
      phi[f] += -Df[f] * ghf[f] * (rho_h[halofid[f]] - rho[iL])


def _smooth_vec_2d(cx: 'float[:]', cy: 'float[:]', cx_h: 'float[:]', cy_h: 'float[:]',
                   cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                   ox: 'float[:]', oy: 'float[:]'):
  # One Laplacian-smoothing pass of a cell vector field: o_P = mean over {P + its
  # face-neighbours}. Smooths the interface normal before the curvature divergence so
  # the discrete curvature is far less noisy (standard VOF fix, Lafaurie et al.).
  n = len(cx); nfc = len(cellid)
  for i in range(n):
    ox[i] = cx[i]; oy[i] = cy[i]
  cnt = np.ones(n)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      ox[iL] += cx[iR]; oy[iL] += cy[iR]; cnt[iL] += 1.0
      ox[iR] += cx[iL]; oy[iR] += cy[iL]; cnt[iR] += 1.0
    elif fname[f] == 10:
      h = halofid[f]
      ox[iL] += cx_h[h]; oy[iL] += cy_h[h]; cnt[iL] += 1.0
  for i in range(n):
    ox[i] /= cnt[i]; oy[i] /= cnt[i]


def _reconstruct_2d(psi: 'float[:]', normal: 'float[:,:]', cellid: 'int64[:,:]',
                    halofid: 'int64[:]', fname: 'int64[:]', ixx: 'float[:]',
                    ixy: 'float[:]', iyy: 'float[:]', rx: 'float[:]', ry: 'float[:]'):
  # Least-squares cell reconstruction of a cell vector U from face-normal fluxes psi_f
  # (U.S_f ~ psi_f): U_P = (sum_f S_f S_f^T)^-1 sum_f S_f psi_f, the inverse metric
  # (ixx,ixy,iyy) precomputed. For a face where P is the neighbour, S_f and psi_f both
  # flip sign, so S_f psi_f is orientation-invariant -> both cells accumulate +S psi.
  # This is OpenFOAM's fvc::reconstruct, used for the well-balanced body-force velocity.
  n = len(rx); nfc = len(cellid)
  for i in range(n):
    rx[i] = 0.0; ry[i] = 0.0
  for f in range(nfc):
    iL = cellid[f, 0]; nx = normal[f, 0]; ny = normal[f, 1]; p = psi[f]
    rx[iL] += nx * p; ry[iL] += ny * p
    if fname[f] == 0:
      iR = cellid[f, 1]
      rx[iR] += nx * p; ry[iR] += ny * p
  for i in range(n):
    bx = rx[i]; by = ry[i]
    rx[i] = ixx[i] * bx + ixy[i] * by
    ry[i] = ixy[i] * bx + iyy[i] * by


_st = {}


def get_vof_st_kernels():
  """Surface-tension kernels: normal smoothing, Green-Gauss divergence (for curvature)
  and the balanced surface-tension face flux."""
  if not _st:
    _st['gg_div'] = compile(_gg_div_2d)
    _st['st_flux'] = compile(_st_face_flux_2d)
    _st['smooth'] = compile(_smooth_vec_2d)
    _st['buoy'] = compile(_buoy_face_flux_2d)
    _st['reconstruct'] = compile(_reconstruct_2d)
  return (_st['gg_div'], _st['st_flux'], _st['smooth'], _st['buoy'], _st['reconstruct'])
