#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-volume compute kernels for the 2D Shallow-Water MHD (SWMHD) system.

Ported from the legacy `manapy/models/SWMHDModel/tools.py` (pyccel) to the new
runtime-compiled backend (`manapy.backends.compile_fun`), following the exact
same structure as `manapy/solvers/shallowater/fvm_utils_compute.py`.

Conserved variables (per cell): h, hu, hv, hB1, hB2, PSI, Z.
The magnetic divergence constraint div(hB) = 0 is handled with a GLM
(generalized Lagrange multiplier) approach through PSI and the cleaning
speed `cpsi`.

NOTHING is compiled at import. Call `setup(dim)` once (uniformly on all MPI
ranks) before using any kernel below; `ShallowWaterMHDSolver` does this in
`__init__`.
"""
import numpy as np
from manapy.backends.compile_fun import compile


def _update_SWMHD(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hB1_c: 'float[:]', hB2_c: 'float[:]',
                  PSI_c: 'float[:]', Z_c: 'float[:]',
                  rez_h: 'float[:]', rez_hu: 'float[:]', rez_hv: 'float[:]', rez_hB1: 'float[:]', rez_hB2: 'float[:]',
                  rez_PSI: 'float[:]', rez_Z: 'float[:]',
                  src_h: 'float[:]', src_hu: 'float[:]', src_hv: 'float[:]', src_hB1: 'float[:]', src_hB2: 'float[:]',
                  src_PSI: 'float[:]', src_Z: 'float[:]',
                  dtime: 'float', cell_volume: 'float[:]', GLM: 'int', cpsi: 'float'):
  for i in range(len(h_c)):
    h_c[i] += dtime * (rez_h[i] + src_h[i]) / cell_volume[i]
    hu_c[i] += dtime * (rez_hu[i] + src_hu[i]) / cell_volume[i]
    hv_c[i] += dtime * (rez_hv[i] + src_hv[i]) / cell_volume[i]
    hB1_c[i] += dtime * (rez_hB1[i] + src_hB1[i]) / cell_volume[i]
    hB2_c[i] += dtime * (rez_hB2[i] + src_hB2[i]) / cell_volume[i]
    PSI_c[i] += dtime * (rez_PSI[i] + src_PSI[i]) / cell_volume[i]
    Z_c[i] += dtime * (rez_Z[i] + src_Z[i]) / cell_volume[i]

  # GLM source-splitting relaxation of PSI (mixed GLM-MHD).
  if GLM == 10:
    cr = 0.01
    for i in range(len(h_c)):
      PSI_c[i] = np.exp(-dtime * (cpsi / cr)) * PSI_c[i]


def _time_step_SWMHD(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hB1_c: 'float[:]', hB2_c: 'float[:]',
                     cfl: 'float', face_normal: 'float[:,:]', face_measure: 'float[:]', cell_volume: 'float[:]',
                     cell_faceid: 'int[:,:]'):
  grav = 1.0
  nbelement = len(cell_faceid)
  u_n = 0.
  B_n = 0.
  dt = 1e6

  for i in range(nbelement):
    lam = 0.
    for j in range(cell_faceid[i][-1]):
      f = cell_faceid[i][j]
      u_n = np.fabs(hu_c[i] / h_c[i] * face_normal[f][0] + hv_c[i] / h_c[i] * face_normal[f][1]) / face_measure[f]
      B_n = np.fabs(hB1_c[i] / h_c[i] * face_normal[f][0] + hB2_c[i] / h_c[i] * face_normal[f][1]) / face_measure[f]
      wb = np.sqrt(B_n ** 2 + grav * h_c[i])
      lam1 = np.fabs(u_n - wb)
      lam2 = np.fabs(u_n - B_n)
      lam3 = np.fabs(u_n + B_n)
      lam4 = np.fabs(u_n + wb)
      lam_convect = max(lam1, lam2, lam3, lam4)
      lam += lam_convect * face_measure[f]
    dt = min(dt, cfl * cell_volume[i] / lam)

  return dt


def _cpsi_global(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hB1_c: 'float[:]', hB2_c: 'float[:]',
                 cfl: 'float', face_normal: 'float[:,:]', face_measure: 'float[:]', cell_volume: 'float[:]',
                 cell_faceid: 'int[:,:]'):
  grav = 1.0
  nbelement = len(cell_faceid)
  u_n = 0.
  B_n = 0.
  cpsiglobal = 0.

  for i in range(nbelement):
    lam = 0.
    lam_convect = 0.
    for j in range(cell_faceid[i][-1]):
      f = cell_faceid[i][j]
      u_n = np.fabs(hu_c[i] / h_c[i] * face_normal[f][0] + hv_c[i] / h_c[i] * face_normal[f][1]) / face_measure[f]
      B_n = np.fabs(hB1_c[i] / h_c[i] * face_normal[f][0] + hB2_c[i] / h_c[i] * face_normal[f][1]) / face_measure[f]
      wb = np.sqrt(B_n ** 2 + grav * h_c[i])
      lam1 = np.fabs(u_n - wb)
      lam2 = np.fabs(u_n - B_n)
      lam3 = np.fabs(u_n + B_n)
      lam4 = np.fabs(u_n + wb)
      lam_convect = max(lam1, lam2, lam3, lam4)
      lam += lam_convect * face_measure[f]
    cpsiglobal = max(cpsiglobal, lam_convect)

  return cpsiglobal


def _term_source_srnh_SWMHD(src_h: 'float[:]', src_hu: 'float[:]', src_hv: 'float[:]', src_hB1: 'float[:]',
                            src_hB2: 'float[:]', src_PSI: 'float[:]', src_Z: 'float[:]',
                            h_c: 'float[:]', Z_c: 'float[:]',
                            h_ghost: 'float[:]', Z_ghost: 'float[:]',
                            h_halo: 'float[:]', Z_halo: 'float[:]',
                            h_x: 'float[:]', h_y: 'float[:]', psi: 'float[:]',
                            hx_halo: 'float[:]', hy_halo: 'float[:]', psi_halo: 'float[:]',
                            cell_nodeid: 'int[:,:]', cell_faceid: 'int[:,:]', cell_cellfid: 'int[:,:]',
                            face_cellid: 'int[:,:]',
                            cell_center: 'float[:,:]', cell_nf: 'float[:,:,:]',
                            face_name: 'int[:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                            nodes: 'float[:,:]', face_haloid: 'int[:]', grav: 'float', order: 'intc'):
  # Well-balanced SRNH topography source (hydro C-property: lake at rest with
  # u=0, h+Z=cst is preserved to machine precision). The magnetic tension lives
  # entirely in the convective flux, as it must for correct dynamics.
  #
  # NB: this scheme is NOT magneto-hydrostatically well-balanced (it does not
  # preserve u=0, h*B_eta=cst, h+Z-B_eta^2/2g=cst over topography). That cannot be
  # done at the source-term level: forcing the source to balance the magnetic
  # tension makes it equal the flux, which removes the tension from the dynamics
  # (verified: exact balancing => spurious flat-bed source = full tension). It
  # needs an effective-surface reconstruction of the interface states (hydrostatic
  # reconstruction, Bouchut-Lhebrard). See NOTES_magnetic_wb.md.
  nbelement = len(h_c)
  hi_p = np.zeros(3)
  zi_p = np.zeros(3)

  zv = np.zeros(3)

  mata = np.zeros(3)
  matb = np.zeros(3)

  ns = np.zeros((3, 3))
  ss = np.zeros((3, 3))
  s_1 = np.zeros(3)
  s_2 = np.zeros(3)
  s_3 = np.zeros(3)
  b = np.zeros(3)
  G = np.zeros(3)

  for i in range(nbelement):

    G[:] = cell_center[i]
    c_1 = 0.
    c_2 = 0.

    for j in range(3):
      f = cell_faceid[i][j]
      ss[j] = cell_nf[i][j]

      if face_name[f] == 10:

        h_1p = h_c[i]
        z_1p = Z_c[i]

        h_p1 = h_halo[face_haloid[f]]
        z_p1 = Z_halo[face_haloid[f]]

      elif face_name[f] == 0 or face_name[f] == 11 or face_name[f] == 22 \
              or face_name[f] == 33 or face_name[f] == 44:

        h_1p = h_c[i]
        z_1p = Z_c[i]

        # Neighbour across face f must be taken from face_cellid (the same
        # neighbour the convective flux uses). Periodic faces (11/22/33/44) are
        # treated like interior faces here: their partner cell is already wired
        # into face_cellid[f][1] by the domain's same-rank periodic pairing, so
        # h/Z of the partner (translation-periodic) feed the well-balanced source
        # instead of an (uninitialised) ghost value.
        if face_cellid[f][0] == i:
          nbr = face_cellid[f][1]
        else:
          nbr = face_cellid[f][0]
        h_p1 = h_c[nbr]
        z_p1 = Z_c[nbr]

      else:
        h_1p = h_c[i]
        z_1p = Z_c[i]

        h_p1 = h_ghost[f]
        z_p1 = Z_ghost[f]

      zv[j] = z_p1
      mata[j] = h_p1 * ss[j][0]
      matb[j] = h_p1 * ss[j][1]

      c_1 = c_1 + (0.5 * (h_1p + h_p1) * 0.5 * (h_1p + h_p1)) * ss[j][0]
      c_2 = c_2 + (0.5 * (h_1p + h_p1) * 0.5 * (h_1p + h_p1)) * ss[j][1]

      hi_p[j] = h_1p
      zi_p[j] = z_1p

    c_3 = 3.0 * h_1p

    delta = (mata[1] * matb[2] - mata[2] * matb[1]) - (mata[0] * matb[2] - matb[0] * mata[2]) + (
              mata[0] * matb[1] - matb[0] * mata[1])

    deltax = c_3 * (mata[1] * matb[2] - mata[2] * matb[1]) - (c_1 * matb[2] - c_2 * mata[2]) + (
              c_1 * matb[1] - c_2 * mata[1])

    deltay = (c_1 * matb[2] - c_2 * mata[2]) - c_3 * (mata[0] * matb[2] - matb[0] * mata[2]) + (
              mata[0] * c_2 - matb[0] * c_1)

    deltaz = (mata[1] * c_2 - matb[1] * c_1) - (mata[0] * c_2 - matb[0] * c_1) + c_3 * (
              mata[0] * matb[1] - matb[0] * mata[1])

    h_1 = deltax / delta
    h_2 = deltay / delta
    h_3 = deltaz / delta

    z_1 = zi_p[0] + hi_p[0] - h_1
    z_2 = zi_p[1] + hi_p[1] - h_2
    z_3 = zi_p[2] + hi_p[2] - h_3

    b[:] = nodes[cell_nodeid[i][1]][0:3]

    ns[0] = np.array([(G[1] - b[1]), -(G[0] - b[0]), 0.])
    ns[1] = ns[0] - ss[1]  # N23
    ns[2] = ns[0] + ss[0]  # N31

    s_1 = 0.5 * h_1 * (zv[0] * ss[0] + z_2 * ns[0] + z_3 * (-1) * ns[2])
    s_2 = 0.5 * h_2 * (zv[1] * ss[1] + z_1 * (-1) * ns[0] + z_3 * ns[1])
    s_3 = 0.5 * h_3 * (zv[2] * ss[2] + z_1 * ns[2] + z_2 * (-1) * ns[1])

    src_h[i] = 0.
    src_hu[i] = -grav * (s_1[0] + s_2[0] + s_3[0])
    src_hv[i] = -grav * (s_1[1] + s_2[1] + s_3[1])
    src_hB1[i] = 0.
    src_hB2[i] = 0.
    src_PSI[i] = 0.
    src_Z[i] = 0.


def _coriolis_source_SWMHD(src_hu: 'float[:]', src_hv: 'float[:]',
                           hu_c: 'float[:]', hv_c: 'float[:]',
                           cell_center: 'float[:,:]', cell_volume: 'float[:]',
                           f0: 'float', beta: 'float', y0: 'float'):
  # Beta-plane Coriolis source, ADDED to the momentum source AFTER the
  # well-balanced topography source (which SETS src_hu/src_hv):
  #   d(hu)/dt += f hv ,  d(hv)/dt += -f hu ,  with f = f0 + beta (y - y0).
  # The meridional gradient df/dy = beta is the restoring mechanism that makes
  # (magneto-)Rossby waves exist; with beta=f0=0 this is a no-op and the solver
  # behaves exactly as before. Purely local (no face reconstruction), so it
  # composes with any mesh/BC.
  #
  # IMPORTANT: the source arrays are VOLUME-INTEGRATED (extensive), exactly like
  # the well-balanced source, because update_SWMHD divides (rez + src) by
  # cell_volume. So the per-volume Coriolis rate f*(hv,-hu) is multiplied by
  # cell_volume here; omitting this makes the source ~1/volume too large.
  for i in range(len(hu_c)):
    f = f0 + beta * (cell_center[i][1] - y0)
    src_hu[i] += cell_volume[i] * f * hv_c[i]
    src_hv[i] += -cell_volume[i] * f * hu_c[i]


def _LF_scheme_MHD(hu_l: 'float', hu_r: 'float', hv_l: 'float', hv_r: 'float', h_l: 'float', h_r: 'float',
                   hB1_l: 'float', hB1_r: 'float', hB2_l: 'float', hB2_r: 'float',
                   Z_l: 'float', Z_r: 'float', normal: 'float[:]', mesure: 'float', grav: 'float', flux: 'float[:]'):
  # Local Lax-Friedrichs (Rusanov) flux for SWMHD (alternative to _srnh_scheme_MHD).
  norm = normal / mesure

  hl = h_l
  hr = h_r
  ul = hu_l / h_l
  vl = hv_l / h_l
  B1l = hB1_l / h_l
  B2l = hB2_l / h_l
  ur = hu_r / h_r
  vr = hv_r / h_r
  B1r = hB1_r / h_r
  B2r = hB2_r / h_r

  n1 = norm[0]
  n2 = norm[1]

  Unl = ul * n1 + vl * n2
  Bnl = B1l * n1 + B2l * n2
  wl = np.sqrt(grav * hl + Bnl ** 2)
  Unr = ur * n1 + vr * n2
  Bnr = B1r * n1 + B2r * n2
  wr = np.sqrt(grav * hr + Bnr ** 2)

  lambda1l = np.fabs(Unl - wl)
  lambda2l = np.fabs(Unl - Bnl)
  lambda3l = np.fabs(Unl + Bnl)
  lambda4l = np.fabs(Unl + wl)

  lambda1r = np.fabs(Unr - wr)
  lambda2r = np.fabs(Unr - Bnr)
  lambda3r = np.fabs(Unr + Bnr)
  lambda4r = np.fabs(Unr + wr)

  ll = max(lambda1l, lambda2l, lambda3l, lambda4l)
  lr = max(lambda1r, lambda2r, lambda3r, lambda4r)

  lambda_star = max(ll, lr)

  q_l = hu_l * norm[0] + hv_l * norm[1]
  m_l = hB1_l * norm[0] + hB2_l * norm[1]
  p_l = 0.5 * grav * h_l * h_l

  q_r = hu_r * norm[0] + hv_r * norm[1]
  m_r = hB1_r * norm[0] + hB2_r * norm[1]
  p_r = 0.5 * grav * h_r * h_r

  fleft_h = q_l
  fleft_hu = q_l * hu_l / h_l + p_l * norm[0] - m_l * hB1_l / h_l
  fleft_hv = q_l * hv_l / h_l + p_l * norm[1] - m_l * hB2_l / h_l
  fleft_hB1 = (hv_l * hB1_l / h_l - hu_l * hB2_l / h_l) * norm[1]
  fleft_hB2 = (hu_l * hB2_l / h_l - hv_l * hB1_l / h_l) * norm[0]

  fright_h = q_r
  fright_hu = q_r * hu_r / h_r + p_r * norm[0] - m_r * hB1_r / h_r
  fright_hv = q_r * hv_r / h_r + p_r * norm[1] - m_r * hB2_r / h_r
  fright_hB1 = (hv_r * hB1_r / h_r - hu_r * hB2_r / h_r) * norm[1]
  fright_hB2 = (hu_r * hB2_r / h_r - hv_r * hB1_r / h_r) * norm[0]

  f_h = 0.5 * (fleft_h + fright_h) - 0.5 * lambda_star * (h_r - h_l)
  f_hu = 0.5 * (fleft_hu + fright_hu) - 0.5 * lambda_star * (hu_r - hu_l)
  f_hv = 0.5 * (fleft_hv + fright_hv) - 0.5 * lambda_star * (hv_r - hv_l)
  f_hB1 = 0.5 * (fleft_hB1 + fright_hB1) - 0.5 * lambda_star * (hB1_r - hB1_l)
  f_hB2 = 0.5 * (fleft_hB2 + fright_hB2) - 0.5 * lambda_star * (hB2_r - hB2_l)

  flux[0] = f_h * mesure
  flux[1] = f_hu * mesure
  flux[2] = f_hv * mesure
  flux[3] = f_hB1 * mesure
  flux[4] = f_hB2 * mesure
  flux[5] = 0.
  flux[6] = 0.


def _srnh_scheme_MHD(hu_l: 'float', hu_r: 'float', hv_l: 'float', hv_r: 'float', h_l: 'float', h_r: 'float',
                     hB1_l: 'float', hB1_r: 'float', hB2_l: 'float', hB2_r: 'float',
                     hPSI_l: 'float', hPSI_r: 'float',
                     Z_l: 'float', Z_r: 'float', normal: 'float[:]', mesure: 'float', grav: 'float',
                     flux: 'float[:]', cpsi: 'float'):
  ninv = np.zeros(2)
  w_dif = np.zeros(6)

  ninv[0] = -1 * normal[1]
  ninv[1] = normal[0]

  u_h = (hu_l / h_l * np.sqrt(h_l) + hu_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))
  v_h = (hv_l / h_l * np.sqrt(h_l) + hv_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))
  B1_h = (hB1_l / h_l * np.sqrt(h_l) + hB1_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))
  B2_h = (hB2_l / h_l * np.sqrt(h_l) + hB2_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))

  un_h = (u_h * normal[0] + v_h * normal[1]) / mesure
  vn_h = (u_h * ninv[0] + v_h * ninv[1]) / mesure
  B1n_h = (B1_h * normal[0] + B2_h * normal[1]) / mesure
  B2n_h = (B1_h * ninv[0] + B2_h * ninv[1]) / mesure

  hroe = (h_l + h_r) / 2
  uroe = un_h
  vroe = vn_h
  B1roe = B1n_h
  B2roe = B2n_h

  uleft = (hu_l * normal[0] + hv_l * normal[1]) / mesure
  vleft = (hu_l * ninv[0] + hv_l * ninv[1]) / mesure
  B1left = (hB1_l * normal[0] + hB2_l * normal[1]) / mesure
  B2left = (hB1_l * ninv[0] + hB2_l * ninv[1]) / mesure

  uright = (hu_r * normal[0] + hv_r * normal[1]) / mesure
  vright = (hu_r * ninv[0] + hv_r * ninv[1]) / mesure
  B1right = (hB1_r * normal[0] + hB2_r * normal[1]) / mesure
  B2right = (hB1_r * ninv[0] + hB2_r * ninv[1]) / mesure

  w_lrh = (h_l + h_r) / 2
  w_lrhu = (uleft + uright) / 2
  w_lrhv = (vleft + vright) / 2
  w_lrhB1 = (B1left + B1right) / 2
  w_lrhB2 = (B2left + B2right) / 2
  w_lrhPSI = (hPSI_l + hPSI_r) / 2
  w_lrz = (Z_l + Z_r) / 2

  w_dif[0] = h_r - h_l
  w_dif[1] = uright - uleft
  w_dif[2] = vright - vleft
  w_dif[3] = B1right - B1left
  w_dif[4] = B2right - B2left
  w_dif[5] = Z_r - Z_l

  signA = np.zeros((6, 6))

  sound = np.sqrt(grav * hroe)

  w = np.sqrt(B1roe * B1roe + grav * hroe)

  lambda1 = uroe - w
  lambda2 = uroe - B1roe
  lambda3 = uroe + B1roe
  lambda4 = uroe + w

  epsilon = 1e-15

  if np.fabs(lambda1) < epsilon:
    s1 = 0.
    pi1 = 0.
  else:
    s1 = lambda1 / np.fabs(lambda1)
    pi1 = s1 / lambda1

  if np.fabs(lambda2) < epsilon:
    s2 = 0.
    pi2 = 0.
  else:
    s2 = lambda2 / np.fabs(lambda2)
    pi2 = 1. / np.fabs(lambda2)

  if np.fabs(lambda3) < epsilon:
    s3 = 0.
    pi3 = 0.
  else:
    s3 = lambda3 / np.fabs(lambda3)
    pi3 = 1. / np.fabs(lambda3)

  if np.fabs(lambda4) < epsilon:
    s4 = 0.
    pi4 = 0.
  else:
    s4 = lambda4 / np.fabs(lambda4)
    pi4 = 1. / np.fabs(lambda4)

  gamma1 = vroe + B2roe
  gamma2 = vroe - B2roe

  sigma1 = vroe * (s1 * lambda4 - s4 * lambda1) - w * (s2 * gamma1 + s3 * gamma2)
  sigma2 = B2roe * (s1 * lambda4 - s4 * lambda1) - w * (s2 * gamma1 - s3 * gamma2)

  if np.fabs(lambda2) < epsilon and np.fabs(lambda3) < epsilon:

    mu1 = B1roe * vroe * pi1 / w - B1roe * vroe * pi4 / w
    mu2 = B1roe * B2roe * pi1 / w - B1roe * B2roe * pi4 / w
    ann = 1

  else:
    mu1 = B1roe * vroe * pi1 / w - B1roe * vroe * pi4 / w - 0.5 * (gamma1 * pi2 - gamma2 * pi3)
    mu2 = B1roe * B2roe * pi1 / w - B1roe * B2roe * pi4 / w - 0.5 * (gamma1 * pi2 + gamma2 * pi3)
    ann = 1

  # 1ere colonne de la matrice A
  signA[0][0] = (s1 * lambda4 - s4 * lambda1) / (2 * w)
  signA[1][0] = lambda1 * lambda4 * (s1 - s4) / (2 * w)
  signA[2][0] = sigma1 / (2 * w)
  signA[3][0] = 0.0
  signA[4][0] = sigma2 / (2 * w)
  signA[5][0] = 0.0

  # 2eme colonne de la matrice A
  signA[0][1] = (s4 - s1) / (2 * w)
  signA[1][1] = (s4 * lambda4 - s1 * lambda1) / (2 * w)
  signA[2][1] = vroe * (s4 - s1) / (2 * w)
  signA[3][1] = 0.0
  signA[4][1] = B2roe * (s4 - s1) / (2 * w)
  signA[5][1] = 0.0

  # 3eme colonne de la matrice A
  signA[0][2] = 0.0
  signA[1][2] = 0.0
  signA[2][2] = (s2 + s3) / 2
  signA[3][2] = 0.0
  signA[4][2] = (s2 - s3) / 2
  signA[5][2] = 0.0

  # 4eme colonne de la matrice A
  signA[0][3] = ann * B1roe * (pi1 - pi4) / w
  signA[1][3] = ann * B1roe * (s1 - s4) / w
  signA[2][3] = ann * mu1
  signA[3][3] = 0.0
  signA[4][3] = ann * mu2
  signA[5][3] = 0.0

  # 5eme colonne de la matrice A
  signA[0][4] = 0.0
  signA[1][4] = 0.0
  signA[2][4] = (s2 - s3) / 2
  signA[3][4] = 0.0
  signA[4][4] = (s2 + s3) / 2
  signA[5][4] = 0.0

  # 6eme colonne de la matrice A
  signA[0][5] = (sound ** 2) * (pi4 - pi1) / (2 * w)
  signA[1][5] = (sound ** 2) * (s4 - s1) / (2 * w)
  signA[2][5] = (sound ** 2) * vroe * (pi4 - pi1) / (2 * w)
  signA[3][5] = 0.0
  signA[4][5] = (sound ** 2) * B2roe * (pi4 - pi1) / (2 * w)
  signA[5][5] = 0.0

  smmat = signA

  hnew = 0.
  unew = 0.
  vnew = 0.
  B1new = 0.
  B2new = 0.
  znew = 0.

  for k in range(6):
    hnew += smmat[0][k] * w_dif[k]
    unew += smmat[1][k] * w_dif[k]
    vnew += smmat[2][k] * w_dif[k]
    B1new += smmat[3][k] * w_dif[k]
    B2new += smmat[4][k] * w_dif[k]
    znew += smmat[5][k] * w_dif[k]

  Pnew = cpsi * (B1right - B1left)
  u_h = hnew / 2
  u_hu = unew / 2
  u_hv = vnew / 2
  u_hP = Pnew / 2
  u_hB1 = B1new / 2
  u_hB2 = B2new / 2
  u_z = znew / 2

  w_lrh = w_lrh - u_h
  w_lrhu = w_lrhu - u_hu
  w_lrhv = w_lrhv - u_hv
  w_lrhP = w_lrhPSI - u_hP
  w_lrhB1 = w_lrhB1 - u_hB1
  w_lrhB2 = w_lrhB2 - u_hB2
  w_lrz = w_lrz - u_z

  w_hP = hPSI_r - hPSI_l
  mw_hB1 = w_lrhB1 / 2 - (1 / (2 * cpsi)) * w_hP
  mhP = (hPSI_r - hPSI_l) / 2 - cpsi * w_dif[3] / 2

  unew = 0.
  vnew = 0.
  B1new = 0.
  B2new = 0.

  unew = (w_lrhu * normal[0] - w_lrhv * normal[1]) / mesure
  vnew = (w_lrhu * normal[1] + w_lrhv * normal[0]) / mesure

  B1new = (w_lrhB1 * normal[0] - w_lrhB2 * normal[1]) / mesure
  B2new = (w_lrhB1 * normal[1] + w_lrhB2 * normal[0]) / mesure

  w_lrhu = unew
  w_lrhv = vnew

  w_lrhB1 = B1new
  w_lrhB2 = B2new

  norm = normal / mesure

  q_s = normal[0] * unew + normal[1] * vnew
  p_s = normal[0] * B1new + normal[1] * B2new

  Flux_B1psi = mhP * norm[0] * mesure
  Flux_B2psi = mhP * norm[1] * mesure
  Flux_hPpsi = cpsi * cpsi * mw_hB1 * mesure

  flux[0] = q_s
  flux[1] = q_s * w_lrhu / w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[0] - p_s * w_lrhB1 / w_lrh
  flux[2] = q_s * w_lrhv / w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[1] - p_s * w_lrhB2 / w_lrh
  flux[3] = (w_lrhv * w_lrhB1 / w_lrh - w_lrhu * w_lrhB2 / w_lrh) * normal[1] + Flux_B1psi
  flux[4] = (w_lrhu * w_lrhB2 / w_lrh - w_lrhv * w_lrhB1 / w_lrh) * normal[0] + Flux_B2psi
  flux[5] = Flux_hPpsi
  flux[6] = 0.


def _explicitscheme_convective_SWMHD(rez_h: 'float[:]', rez_hu: 'float[:]', rez_hv: 'float[:]', rez_hB1: 'float[:]',
                                     rez_hB2: 'float[:]', rez_PSI: 'float[:]', rez_Z: 'float[:]',
                                     h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hB1_c: 'float[:]',
                                     hB2_c: 'float[:]', hPSIc: 'float[:]', Z_c: 'float[:]',
                                     h_ghost: 'float[:]', hu_ghost: 'float[:]', hv_ghost: 'float[:]',
                                     hB1_ghost: 'float[:]', hB2_ghost: 'float[:]', hPSIghost: 'float[:]',
                                     Z_ghost: 'float[:]',
                                     h_halo: 'float[:]', hu_halo: 'float[:]', hv_halo: 'float[:]', hB1_halo: 'float[:]',
                                     hB2_halo: 'float[:]', hPSIhalo: 'float[:]', Z_halo: 'float[:]',
                                     h_x: 'float[:]', h_y: 'float[:]', hx_halo: 'float[:]', hy_halo: 'float[:]',
                                     psi: 'float[:]', psi_halo: 'float[:]',
                                     cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                     face_cellid: 'int[:,:]', face_measure: 'float[:]', face_normal: 'float[:,:]',
                                     face_haloid: 'int[:]',
                                     d_innerfaces: 'int[:]', d_halofaces: 'int[:]', d_boundaryfaces: 'int[:]',
                                     d_periodicboundaryfaces: 'int[:]',
                                     grav: 'float', order: 'int', cpsi: 'float'):
  rez_h[:] = 0.
  rez_hu[:] = 0.
  rez_hv[:] = 0.
  rez_hB1[:] = 0.
  rez_hB2[:] = 0.
  rez_PSI[:] = 0.
  rez_Z[:] = 0.

  flux = np.zeros(7)

  for i in d_innerfaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hB1_l = hB1_c[face_cellid[i][0]]
    hB2_l = hB2_c[face_cellid[i][0]]
    hPSI_l = hPSIc[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]

    h_r = h_c[face_cellid[i][1]]
    hu_r = hu_c[face_cellid[i][1]]
    hv_r = hv_c[face_cellid[i][1]]
    hB1_r = hB1_c[face_cellid[i][1]]
    hB2_r = hB2_c[face_cellid[i][1]]
    hPSI_r = hPSIc[face_cellid[i][1]]
    Z_r = Z_c[face_cellid[i][1]]

    _srnh_scheme_MHD(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hB1_l, hB1_r, hB2_l, hB2_r, hPSI_l, hPSI_r, Z_l, Z_r,
                     normal, mesure, grav, flux, cpsi)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hB1[face_cellid[i][0]] -= flux[3]
    rez_hB2[face_cellid[i][0]] -= flux[4]
    rez_PSI[face_cellid[i][0]] -= flux[5]
    rez_Z[face_cellid[i][0]] -= flux[6]

    rez_h[face_cellid[i][1]] += flux[0]
    rez_hu[face_cellid[i][1]] += flux[1]
    rez_hv[face_cellid[i][1]] += flux[2]
    rez_hB1[face_cellid[i][1]] += flux[3]
    rez_hB2[face_cellid[i][1]] += flux[4]
    rez_PSI[face_cellid[i][1]] += flux[5]
    rez_Z[face_cellid[i][1]] += flux[6]

  for i in d_halofaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hB1_l = hB1_c[face_cellid[i][0]]
    hB2_l = hB2_c[face_cellid[i][0]]
    hPSI_l = hPSIc[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]
    h_r = h_halo[face_haloid[i]]
    hu_r = hu_halo[face_haloid[i]]
    hv_r = hv_halo[face_haloid[i]]
    hB1_r = hB1_halo[face_haloid[i]]
    hB2_r = hB2_halo[face_haloid[i]]
    hPSI_r = hPSIhalo[face_haloid[i]]
    Z_r = Z_halo[face_haloid[i]]

    _srnh_scheme_MHD(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hB1_l, hB1_r, hB2_l, hB2_r, hPSI_l, hPSI_r, Z_l, Z_r,
                     normal, mesure, grav, flux, cpsi)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hB1[face_cellid[i][0]] -= flux[3]
    rez_hB2[face_cellid[i][0]] -= flux[4]
    rez_PSI[face_cellid[i][0]] -= flux[5]
    rez_Z[face_cellid[i][0]] -= flux[6]

  for i in d_periodicboundaryfaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hB1_l = hB1_c[face_cellid[i][0]]
    hB2_l = hB2_c[face_cellid[i][0]]
    hPSI_l = hPSIc[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]

    h_r = h_c[face_cellid[i][1]]
    hu_r = hu_c[face_cellid[i][1]]
    hv_r = hv_c[face_cellid[i][1]]
    hB1_r = hB1_c[face_cellid[i][1]]
    hB2_r = hB2_c[face_cellid[i][1]]
    hPSI_r = hPSIc[face_cellid[i][1]]
    Z_r = Z_c[face_cellid[i][1]]

    _srnh_scheme_MHD(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hB1_l, hB1_r, hB2_l, hB2_r, hPSI_l, hPSI_r, Z_l, Z_r,
                     normal, mesure, grav, flux, cpsi)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hB1[face_cellid[i][0]] -= flux[3]
    rez_hB2[face_cellid[i][0]] -= flux[4]
    rez_PSI[face_cellid[i][0]] -= flux[5]
    rez_Z[face_cellid[i][0]] -= flux[6]

  for i in d_boundaryfaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hB1_l = hB1_c[face_cellid[i][0]]
    hB2_l = hB2_c[face_cellid[i][0]]
    hPSI_l = hPSIc[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]
    h_r = h_ghost[i]
    hu_r = hu_ghost[i]
    hv_r = hv_ghost[i]
    hB1_r = hB1_ghost[i]
    hB2_r = hB2_ghost[i]
    hPSI_r = hPSIghost[i]
    Z_r = Z_ghost[i]

    _srnh_scheme_MHD(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hB1_l, hB1_r, hB2_l, hB2_r, hPSI_l, hPSI_r, Z_l, Z_r,
                     normal, mesure, grav, flux, cpsi)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hB1[face_cellid[i][0]] -= flux[3]
    rez_hB2[face_cellid[i][0]] -= flux[4]
    rez_PSI[face_cellid[i][0]] -= flux[5]
    rez_Z[face_cellid[i][0]] -= flux[6]


############################################################################
# NOTHING is compiled at import. Call setup(dim) once (uniformly on all MPI
# ranks) before using any kernel above; ShallowWaterMHDSolver does this in
# __init__. The SWMHD kernels are dimension-agnostic (2D triangular meshes),
# so they are compiled once. The nested helper _srnh_scheme_MHD is compiled
# (and rebound to the module global) before the kernel that calls it, so numba
# can resolve the njit->njit call.
_agnostic_done = False


def setup(dim):
  global _agnostic_done
  if dim not in (2, 3):
    raise ValueError(f"Unsupported dimension: {dim}")
  if not _agnostic_done:
    global _srnh_scheme_MHD, _LF_scheme_MHD  # nested helpers first
    global update_SWMHD, time_step_SWMHD, cpsi_global, term_source_srnh_SWMHD
    global explicitscheme_convective_SWMHD, coriolis_source_SWMHD
    _srnh_scheme_MHD = compile(_srnh_scheme_MHD)
    _LF_scheme_MHD = compile(_LF_scheme_MHD)
    update_SWMHD = compile(_update_SWMHD)
    time_step_SWMHD = compile(_time_step_SWMHD)
    cpsi_global = compile(_cpsi_global)
    term_source_srnh_SWMHD = compile(_term_source_srnh_SWMHD)
    explicitscheme_convective_SWMHD = compile(_explicitscheme_convective_SWMHD)
    coriolis_source_SWMHD = compile(_coriolis_source_SWMHD)
    _agnostic_done = True
