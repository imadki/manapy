#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-epsilon RANS turbulence closure for the 2D Shallow-Water MHD system.

Implements the model of "RANS modeling of shallow water magnetohydrodynamics"
(Favre-averaged SWMHD, dual kinetic+magnetic turbulent energy/dissipation):

  variables (per cell, Favre-averaged, height-weighted h*.):
    kc = h*k1  turbulent KINETIC energy (density)   -> nu_t = Cmu kc^2 / (h epsc)
    km = h*k2  turbulent MAGNETIC energy (density)   -> mu_t = Cmu km^2 / (h epsm)
    epsc = h*eps1  kinetic dissipation
    epsm = h*eps2  magnetic dissipation

Each is transported by  d(h k)/dt + div(h k u) = div(Gamma grad k) + S ,
with a spatially-VARYING diffusion coefficient Gamma = (nu + nu_t/sigma) h.
The eddy viscosities nu_t, mu_t feed back into the SWMHD momentum (turbulent
stress) and induction (turbulent resistivity) equations.

These kernels reuse the manapy FV machinery: convective transport via the SRNH
flux (system.py), face gradients via Variable.compute_face_gradient(), and the
diffusion operator below is the variable-coefficient generalisation of
solvers/advecdiff._explicitscheme_dissipative (scalar Dxx -> face array Gamma).

NOTHING is compiled at import; call setup() once (as ShallowWaterMHDSolver does).
"""
import numpy as np
from manapy.backends.compile_fun import compile


def _eddy_viscosity(h_c: 'float[:]', kc: 'float[:]', epsc: 'float[:]', km: 'float[:]', epsm: 'float[:]',
                    nu_t: 'float[:]', mu_t: 'float[:]',
                    Cmu: 'float', k_floor: 'float', eps_floor: 'float'):
  # nu_t = Cmu * k_c^2 / eps_c  and  mu_t = Cmu * k_m^2 / eps_m, computed from the
  # per-mass Favre quantities k = (h k)/h, eps = (h eps)/h, with realizability
  # floors so that k,eps>0 keep nu_t,mu_t finite and non-negative.
  for i in range(len(h_c)):
    h = h_c[i]
    if h < 1e-12:
      nu_t[i] = 0.0
      mu_t[i] = 0.0
      continue
    kc_i = kc[i] / h
    ec_i = epsc[i] / h
    km_i = km[i] / h
    em_i = epsm[i] / h
    if kc_i < k_floor:
      kc_i = k_floor
    if ec_i < eps_floor:
      ec_i = eps_floor
    if km_i < k_floor:
      km_i = k_floor
    if em_i < eps_floor:
      em_i = eps_floor
    nu_t[i] = Cmu * kc_i * kc_i / ec_i
    mu_t[i] = Cmu * km_i * km_i / em_i


def _cell_to_face_coef(gam_c: 'float[:]', gam_ghost: 'float[:]', gam_halo: 'float[:]',
                       face_cellid: 'int[:,:]', face_name: 'int[:]', face_haloid: 'int[:]',
                       gam_face: 'float[:]'):
  # Arithmetic-mean interpolation of a cell diffusion coefficient to the faces.
  # face_name: 0 = inner, 10 = halo, else = physical boundary (ghost).
  for i in range(len(face_cellid)):
    nm = face_name[i]
    if nm == 0:
      gam_face[i] = 0.5 * (gam_c[face_cellid[i][0]] + gam_c[face_cellid[i][1]])
    elif nm == 10:
      gam_face[i] = 0.5 * (gam_c[face_cellid[i][0]] + gam_halo[face_haloid[i]])
    else:
      gam_face[i] = 0.5 * (gam_c[face_cellid[i][0]] + gam_ghost[i])


def _diffusion_varcoef(wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]',
                       gam_face: 'float[:]', face_cellid: 'int[:,:]', face_normal: 'float[:,:]',
                       face_name: 'int[:]', dissip_w: 'float[:]'):
  # div(Gamma grad w) with a face-varying coefficient Gamma = gam_face.
  # Generalises solvers/advecdiff._explicitscheme_dissipative (constant Dxx,Dyy,Dzz)
  # to a per-face coefficient; identical to it when gam_face is constant and
  # isotropic. Accumulates the FV flux per cell (interior faces to both sides).
  nbface = len(face_cellid)
  norm = np.zeros(3)
  dissip_w[:] = 0.
  for i in range(nbface):
    norm[:] = face_normal[i][:]
    q = gam_face[i] * (wx_face[i] * norm[0] + wy_face[i] * norm[1] + wz_face[i] * norm[2])
    if face_name[i] == 0:
      dissip_w[face_cellid[i][0]] += q
      dissip_w[face_cellid[i][1]] -= q
    else:
      dissip_w[face_cellid[i][0]] += q


def _turbulence_source(h_c: 'float[:]', kc: 'float[:]', km: 'float[:]', epsc: 'float[:]', epsm: 'float[:]',
                       nu_t: 'float[:]', mu_t: 'float[:]',
                       dudx: 'float[:]', dudy: 'float[:]', dvdx: 'float[:]', dvdy: 'float[:]',
                       dB1dx: 'float[:]', dB1dy: 'float[:]', dB2dx: 'float[:]', dB2dy: 'float[:]',
                       src_kc: 'float[:]', src_km: 'float[:]', src_epsc: 'float[:]', src_epsm: 'float[:]',
                       Cmu: 'float', Ce1: 'float', Ce2: 'float', be1: 'float', be2: 'float',
                       Ce3: 'float', be3: 'float', sigma_k: 'float', k_floor: 'float', eps_floor: 'float'):
  # Algebraic source of the dual k-epsilon transport (RANS-SWMHD, eq. 2.13-2.14).
  # Density form kc=h k_c, epsc=h eps_c, ... :
  #   d(h k_c)/dt + adv = diff + h ( P^u_k + R_k1            - eps_c )
  #   d(h k_m)/dt + adv = diff + h ( -P^B_k - R_k2           - eps_m )
  #   d(h eps_c)/dt     = ... + h ( (eps_c/k_c)(Ce1 P^u_k + Ce3 R_k1) - Ce2 eps_c^2/k_c )
  #   d(h eps_m)/dt     = ... + h ( (eps_m/k_m)(-be1 P^B_k - be3 R_k2) - be2 eps_m^2/k_m )
  # productions and kinetic<->magnetic exchanges (nu_t/sigma_k EMF closure):
  #   P^u_k = nu_t [2 ux^2 + 2 vy^2 + (uy+vx)^2]
  #   P^B_k = mu_t [2 B1x ux + 2 B2y vy + (B1y+B2x)(uy+vx)]
  #   R_k1  = -(nu_t/sigma_k)(B1x^2 + 2 B1y B2x + B2y^2)          (grad B~ : transpose)
  #   R_k2  = -(nu_t/sigma_k)(B1x^2 + B1y^2 + B2x^2 + B2y^2)      (-|grad B~|^2)
  # NOTE: the transport/higher-derivative cross terms T_k, D_k (eq. 2.13) are still
  # omitted: T_k ~ B~.grad(nu_t/sigma_k div B~) ~ 0 under GLM div-cleaning, and D_k is
  # a 4th-order term small vs the algebraic exchanges kept here.
  for i in range(len(h_c)):
    h = h_c[i]
    if h < 1e-12:
      src_kc[i] = 0.0; src_km[i] = 0.0; src_epsc[i] = 0.0; src_epsm[i] = 0.0
      continue
    k_c = kc[i] / h; e_c = epsc[i] / h
    k_m = km[i] / h; e_m = epsm[i] / h
    if k_c < k_floor:
      k_c = k_floor
    if e_c < eps_floor:
      e_c = eps_floor
    if k_m < k_floor:
      k_m = k_floor
    if e_m < eps_floor:
      e_m = eps_floor
    ux = dudx[i]; uy = dudy[i]; vx = dvdx[i]; vy = dvdy[i]
    B1x = dB1dx[i]; B1y = dB1dy[i]; B2x = dB2dx[i]; B2y = dB2dy[i]
    Pu = nu_t[i] * (2.0 * ux * ux + 2.0 * vy * vy + (uy + vx) * (uy + vx))
    Pb = mu_t[i] * (2.0 * B1x * ux + 2.0 * B2y * vy + (B1y + B2x) * (uy + vx))
    nts = nu_t[i] / sigma_k
    Rk1 = -nts * (B1x * B1x + 2.0 * B1y * B2x + B2y * B2y)
    Rk2 = -nts * (B1x * B1x + B1y * B1y + B2x * B2x + B2y * B2y)
    # densities (per-mass rates x h)
    src_kc[i] = h * (Pu + Rk1 - e_c)
    src_km[i] = h * (-Pb - Rk2 - e_m)
    src_epsc[i] = h * (e_c / k_c) * (Ce1 * Pu + Ce3 * Rk1 - Ce2 * e_c)
    src_epsm[i] = h * (e_m / k_m) * (-be1 * Pb - be3 * Rk2 - be2 * e_m)


def _advect_density(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                    u_c: 'float[:]', v_c: 'float[:]', u_ghost: 'float[:]', v_ghost: 'float[:]',
                    u_halo: 'float[:]', v_halo: 'float[:]',
                    face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
                    face_haloid: 'int[:]', rez_w: 'float[:]'):
  # First-order upwind FV transport of a density w (= h*k):  d w/dt + div(w u) = rez_w.
  # Face velocity = arithmetic mean of the two sides; upwind selection on u.n.
  # face_name: 0 inner, 10 halo, else physical boundary (ghost).
  rez_w[:] = 0.
  for i in range(len(face_cellid)):
    L = face_cellid[i][0]
    nx = face_normal[i][0]; ny = face_normal[i][1]
    nm = face_name[i]
    if nm == 0:
      R = face_cellid[i][1]
      un = 0.5 * (u_c[L] + u_c[R]) * nx + 0.5 * (v_c[L] + v_c[R]) * ny
      wup = w_c[L] if un >= 0.0 else w_c[R]
      flux = un * wup
      rez_w[L] -= flux
      rez_w[R] += flux
    elif nm == 10:
      k = face_haloid[i]
      un = 0.5 * (u_c[L] + u_halo[k]) * nx + 0.5 * (v_c[L] + v_halo[k]) * ny
      wup = w_c[L] if un >= 0.0 else w_halo[k]
      rez_w[L] -= un * wup
    else:
      un = 0.5 * (u_c[L] + u_ghost[i]) * nx + 0.5 * (v_c[L] + v_ghost[i]) * ny
      wup = w_c[L] if un >= 0.0 else w_ghost[i]
      rez_w[L] -= un * wup


def _stress_divergence_face(uxf: 'float[:]', uyf: 'float[:]', vxf: 'float[:]', vyf: 'float[:]',
                            B1xf: 'float[:]', B1yf: 'float[:]', B2xf: 'float[:]', B2yf: 'float[:]',
                            nut_f: 'float[:]', mut_f: 'float[:]', h_f: 'float[:]', nu: 'float', mu: 'float',
                            face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
                            rez_hu: 'float[:]', rez_hv: 'float[:]'):
  # Full anisotropic turbulent stress divergence, evaluated DIRECTLY at faces from
  # face gradients (exact for linear fields, unlike a cell-gradient + interpolation
  # path). tau built per face; flux (tau.n) accumulated into the momentum residuals.
  #   txx = (nu+2 nu_t) h u~_x - 2 mu_t h B1~_x
  #   txy = (nu+nu_t) h (u~_y+v~_x) - mu_t h (B1~_y+B2~_x)
  #   tyy = (nu+2 nu_t) h v~_y - 2 mu_t h B2~_y
  rez_hu[:] = 0.
  rez_hv[:] = 0.
  for i in range(len(face_cellid)):
    nx = face_normal[i][0]; ny = face_normal[i][1]
    nt = nut_f[i]; mt = mut_f[i]; hh = h_f[i]
    txx = (nu + 2.0 * nt) * hh * uxf[i] - 2.0 * mt * hh * B1xf[i]
    txy = (nu + nt) * hh * (uyf[i] + vxf[i]) - mt * hh * (B1yf[i] + B2xf[i])
    tyy = (nu + 2.0 * nt) * hh * vyf[i] - 2.0 * mt * hh * B2yf[i]
    fu = txx * nx + txy * ny
    fv = txy * nx + tyy * ny
    if face_name[i] == 0:
      L = face_cellid[i][0]; R = face_cellid[i][1]
      rez_hu[L] += fu; rez_hu[R] -= fu
      rez_hv[L] += fv; rez_hv[R] -= fv
    else:
      rez_hu[face_cellid[i][0]] += fu
      rez_hv[face_cellid[i][0]] += fv


def _stress_divergence(txx_c: 'float[:]', txy_c: 'float[:]', tyy_c: 'float[:]',
                       txx_g: 'float[:]', txy_g: 'float[:]', tyy_g: 'float[:]',
                       txx_h: 'float[:]', txy_h: 'float[:]', tyy_h: 'float[:]',
                       face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
                       face_haloid: 'int[:]', rez_hu: 'float[:]', rez_hv: 'float[:]'):
  # Divergence of a symmetric stress tensor tau = [[txx,txy],[txy,tyy]], ACCUMULATED
  # into the momentum residuals:  rez_hu += div(txx,txy),  rez_hv += div(txy,tyy).
  # tau is interpolated to faces (arithmetic mean). This gives the full anisotropic
  # turbulent stress S_u,S_v of the RANS-SWMHD model when
  #   txx = (nu+2 nu_t) h u~_x - 2 mu_t h B1~_x
  #   txy = (nu+nu_t) h (u~_y+v~_x) - mu_t h (B1~_y+B2~_x)
  #   tyy = (nu+2 nu_t) h v~_y - 2 mu_t h B2~_y
  rez_hu[:] = 0.
  rez_hv[:] = 0.
  for i in range(len(face_cellid)):
    L = face_cellid[i][0]
    nx = face_normal[i][0]; ny = face_normal[i][1]
    nm = face_name[i]
    if nm == 0:
      R = face_cellid[i][1]
      xx = 0.5 * (txx_c[L] + txx_c[R]); xy = 0.5 * (txy_c[L] + txy_c[R]); yy = 0.5 * (tyy_c[L] + tyy_c[R])
      fu = xx * nx + xy * ny; fv = xy * nx + yy * ny
      rez_hu[L] += fu; rez_hu[R] -= fu
      rez_hv[L] += fv; rez_hv[R] -= fv
    elif nm == 10:
      k = face_haloid[i]
      xx = 0.5 * (txx_c[L] + txx_h[k]); xy = 0.5 * (txy_c[L] + txy_h[k]); yy = 0.5 * (tyy_c[L] + tyy_h[k])
      rez_hu[L] += xx * nx + xy * ny; rez_hv[L] += xy * nx + yy * ny
    else:
      xx = 0.5 * (txx_c[L] + txx_g[i]); xy = 0.5 * (txy_c[L] + txy_g[i]); yy = 0.5 * (tyy_c[L] + tyy_g[i])
      rez_hu[L] += xx * nx + xy * ny; rez_hv[L] += xy * nx + yy * ny


def _stress_tensor(h_c: 'float[:]', nu_t: 'float[:]', mu_t: 'float[:]', nu: 'float', mu: 'float',
                   ux: 'float[:]', uy: 'float[:]', vx: 'float[:]', vy: 'float[:]',
                   B1x: 'float[:]', B1y: 'float[:]', B2x: 'float[:]', B2y: 'float[:]',
                   txx: 'float[:]', txy: 'float[:]', tyy: 'float[:]'):
  # Per-cell symmetric turbulent stress tensor (kinetic Newtonian part + Maxwell
  # magnetic part), height-weighted, as in S_u,S_v of the RANS-SWMHD model.
  for i in range(len(h_c)):
    h = h_c[i]; nt = nu_t[i]; mt = mu_t[i]
    txx[i] = (nu + 2.0 * nt) * h * ux[i] - 2.0 * mt * h * B1x[i]
    txy[i] = (nu + nt) * h * (uy[i] + vx[i]) - mt * h * (B1y[i] + B2x[i])
    tyy[i] = (nu + 2.0 * nt) * h * vy[i] - 2.0 * mt * h * B2y[i]


_done = False


def setup():
  global _done, eddy_viscosity, cell_to_face_coef, diffusion_varcoef, turbulence_source
  global advect_density, stress_divergence, stress_divergence_face, stress_tensor
  if not _done:
    eddy_viscosity = compile(_eddy_viscosity)
    cell_to_face_coef = compile(_cell_to_face_coef)
    diffusion_varcoef = compile(_diffusion_varcoef)
    turbulence_source = compile(_turbulence_source)
    advect_density = compile(_advect_density)
    stress_divergence = compile(_stress_divergence)
    stress_divergence_face = compile(_stress_divergence_face)
    stress_tensor = compile(_stress_tensor)
    _done = True
