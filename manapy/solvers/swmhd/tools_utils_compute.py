#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialisation and diagnostic kernels for the 2D SWMHD system.

Ported from the legacy `manapy/models/SWMHDModel/tools.py`. The public objects
below are `FunObj` wrappers: each is compiled the first time it is called.
"""
import numpy as np
from manapy.backends.compile_fun import FunObj


def _initialisation_SWMHD(h: 'float[:]', hu: 'float[:]', hv: 'float[:]', hB1: 'float[:]', hB2: 'float[:]',
                          PSI: 'float[:]', Z: 'float[:]', center: 'float[:,:]', choix: 'int',
                          k1: 'float', k2: 'float', eps: 'float', tol: 'float'):
  nbelements = len(center)

  if choix == 1:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      hint = 1. / 4
      uint = 1 + 0.5 * np.sin(np.pi * ycent) + hint * np.cos(np.pi * xcent)
      vint = 1 + hint * np.sin(np.pi * xcent) + 0.5 * np.cos(np.pi * ycent)

      h[i] = hint
      hu[i] = hint * uint
      hv[i] = hint * vint
      hB1[i] = hint * (0.5)
      hB2[i] = hint * uint

  if choix == 7:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      AA = np.sin(2 * np.pi * xcent - 2 * np.pi * ycent)

      uu = 2.
      bb = 5.
      h0 = 100.
      u0 = (uu + AA) / h0
      v0 = (uu + AA) / h0
      B01 = (bb + 2 * AA) / h0
      B02 = (bb + 2 * AA) / h0

      PSI[i] = 0.
      h[i] = 100
      hu[i] = uu + AA
      hv[i] = uu + AA
      hB1[i] = bb + 2 * AA
      hB2[i] = bb + 2 * AA
      PSI[i] = 0.

      grav = 1.0
      k = np.sqrt(k1 ** 2 + k2 ** 2)
      c0 = np.sqrt(h0 * grav)
      # Produit scalaire de (k1,k2) et (B01, B02)
      ss = k1 * B01 + k2 * B02
      a1 = np.sqrt(ss ** 2 + (k * c0) ** 2)
      b1 = - 0.5 * eps * k * k

      h_hot = h0 * k * k * np.cos(k1 * xcent + k2 * ycent)
      u_hot = k1 * (a1 * np.cos(k1 * xcent + k2 * ycent) - b1 * np.sin(k1 * xcent + k2 * ycent))
      v_hot = k2 * (a1 * np.cos(k1 * xcent + k2 * ycent) - b1 * np.sin(k1 * xcent + k2 * ycent))
      B1_hot = -k1 * ss * np.cos(k1 * xcent + k2 * ycent)
      B2_hot = -k2 * ss * np.cos(k1 * xcent + k2 * ycent)

      hint = h0 + tol * h_hot
      uint = u0 + tol * u_hot
      vint = v0 + tol * v_hot
      B1int = B01 + tol * B1_hot
      B2int = B02 + tol * B2_hot

      h[i] = hint
      hu[i] = hint * uint
      hv[i] = hint * vint
      hB1[i] = hint * B1int
      hB2[i] = hint * B2int

  if choix == -3:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      AA = 0.0 * np.sin(2 * np.pi * xcent - 2 * np.pi * ycent)

      uu = 0.
      bb = 3.
      h0 = 2.
      u0 = (uu + AA) / h0
      v0 = (uu + AA) / h0
      B01 = (bb + 2 * AA) / h0
      B02 = (bb + 2 * AA) / h0

      PSI[i] = 0.
      h[i] = 100
      hu[i] = uu + AA
      hv[i] = uu + AA
      hB1[i] = bb + 2 * AA
      hB2[i] = bb + 2 * AA
      PSI[i] = 0.

      grav = 1.0
      k = np.sqrt(k1 ** 2 + k2 ** 2)
      c0 = np.sqrt(h0 * grav)
      ss = k1 * B01 + k2 * B02
      a1 = np.sqrt(ss ** 2 + (k * c0) ** 2)
      b1 = - 0.5 * eps * k * k

      h_hot = h0 * k * k * np.cos(k1 * xcent + k2 * ycent)
      u_hot = k1 * (a1 * np.cos(k1 * xcent + k2 * ycent) - b1 * np.sin(k1 * xcent + k2 * ycent))
      v_hot = k2 * (a1 * np.cos(k1 * xcent + k2 * ycent) - b1 * np.sin(k1 * xcent + k2 * ycent))
      B1_hot = -k1 * ss * np.cos(k1 * xcent + k2 * ycent)
      B2_hot = -k2 * ss * np.cos(k1 * xcent + k2 * ycent)

      hint = h0 + tol * h_hot
      uint = u0 + tol * u_hot
      vint = v0 + tol * v_hot
      B1int = B01 + tol * B1_hot
      B2int = B02 + tol * B2_hot

      h[i] = hint
      hu[i] = hint * uint
      hv[i] = hint * vint
      hB1[i] = hint * B1int
      hB2[i] = hint * B2int

  if choix == -2:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      AA = np.sin(2 * np.pi * xcent - 2 * np.pi * ycent)

      uu = 2.
      bb = 5.
      h0 = 100.
      u0 = (uu + AA) / h0
      v0 = (uu + AA) / h0
      B01 = (bb + 2 * AA) / h0
      B02 = (bb + 2 * AA) / h0

      PSI[i] = 0.
      h[i] = 100
      hu[i] = uu + AA
      hv[i] = uu + AA
      hB1[i] = bb + 2 * AA
      hB2[i] = bb + 2 * AA
      PSI[i] = 0.

      grav = 1.0
      k = np.sqrt(k1 ** 2 + k2 ** 2)
      c0 = np.sqrt(h0 * grav)
      ss = k1 * B01 + k2 * B02
      b1 = - 0.5 * eps * k * k

      h_hot = 0.
      u_hot = k2 * ss * np.cos(k1 * xcent + k2 * ycent)
      v_hot = -k1 * ss * np.cos(k1 * xcent + k2 * ycent)
      B1_hot = -k2 * (ss * np.cos(k1 * xcent + k2 * ycent) + b1 * np.sin(k1 * xcent + k2 * ycent))
      B2_hot = -k1 * (ss * np.cos(k1 * xcent + k2 * ycent) + b1 * np.sin(k1 * xcent + k2 * ycent))

      hint = h0 + tol * h_hot
      uint = u0 + tol * u_hot
      vint = v0 + tol * v_hot
      B1int = B01 + tol * B1_hot
      B2int = B02 + tol * B2_hot

      h[i] = hint
      hu[i] = hint * uint
      hv[i] = hint * vint
      hB1[i] = hint * B1int
      hB2[i] = hint * B2int

  elif choix == 40:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      g = 1.0
      umax = 0.2
      Bmax = 0.1
      hmax = 1.0

      rcent = np.sqrt(xcent ** 2 + ycent ** 2)
      ee = np.exp(1 - rcent ** 2)
      e1 = np.exp(0.5 * (1 - rcent ** 2))
      hin = hmax - (1. / (2.0 * g)) * (umax ** 2 - Bmax ** 2) * ee
      uin = 1.0 - umax * e1 * ycent
      vin = 1.0 + umax * e1 * xcent
      B1in = -Bmax * e1 * ycent
      B2in = Bmax * e1 * xcent

      h[i] = hin
      hu[i] = hin * uin
      hv[i] = hin * vin
      hB1[i] = hin * B1in
      hB2[i] = hin * B2in
      PSI[i] = 0.

  elif choix == 2:

    # 1D dam break (Cissé et al., ddm choix=2). Domain split at x=0.
    for i in range(nbelements):
      xcent = center[i][0]
      if xcent <= 0:
        h[i] = 1.
        hu[i] = 0.
        hv[i] = 0.
        hB1[i] = 1.
        hB2[i] = 0.
        Z[i] = 0.
      else:
        h[i] = 2.
        hu[i] = 0.
        hv[i] = 0.
        hB1[i] = 1.
        hB2[i] = 2.
        Z[i] = 0.
      PSI[i] = 0.

  elif choix == 3:

    # 2D explosion (Cissé et al., ddm choix=3). Disc r<=0.3.
    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]
      if np.sqrt(xcent ** 2 + ycent ** 2) <= 0.3:
        h[i] = 1.
      else:
        h[i] = 0.1
      hu[i] = 0.
      hv[i] = 0.
      hB1[i] = 0.1
      hB2[i] = 0.0
      Z[i] = 0.
      PSI[i] = 0.

  elif choix == 5:

    # C-property, lake at rest over a gaussian bump, NO magnetic field
    # (Cissé et al., ddm choix=5). h + Z = 1, u = v = B = 0.
    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]
      Z[i] = 0.8 * np.exp(-5 * (xcent - 1) ** 2 - 50 * (ycent - 0.5) ** 2)
      h[i] = 1. - Z[i]
      hu[i] = 0.
      hv[i] = 0.
      hB1[i] = 0.
      hB2[i] = 0.
      PSI[i] = 0.

  elif choix == 6:

    # C-property WITH magnetic field (Cissé et al., ddm choix=6).
    # h + Z = 1, u = v = 0, (B1, B2) = (1, 1e-4) constant.
    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]
      Z[i] = 0.8 * np.exp(-5 * (xcent - 1) ** 2 - 50 * (ycent - 0.5) ** 2)
      h[i] = 1. - Z[i]
      hu[i] = 0.
      hv[i] = 0.
      hB1[i] = 1.
      hB2[i] = 1e-4
      PSI[i] = 0.

  elif choix == 41:

    # Same rotating MHD vortex as choix == 40, but re-centred on (xc, yc) so it
    # fits a unit square [0,1]x[0,1] mesh (choix == 40 is centred on the origin).
    xc = 0.5
    yc = 0.5
    for i in range(nbelements):
      xcent = center[i][0] - xc
      ycent = center[i][1] - yc

      g = 1.0
      umax = 0.2
      Bmax = 0.1
      hmax = 1.0

      rcent = np.sqrt(xcent ** 2 + ycent ** 2)
      ee = np.exp(1 - rcent ** 2)
      e1 = np.exp(0.5 * (1 - rcent ** 2))
      hin = hmax - (1. / (2.0 * g)) * (umax ** 2 - Bmax ** 2) * ee
      uin = 1.0 - umax * e1 * ycent
      vin = 1.0 + umax * e1 * xcent
      B1in = -Bmax * e1 * ycent
      B2in = Bmax * e1 * xcent

      h[i] = hin
      hu[i] = hin * uin
      hv[i] = hin * vin
      hB1[i] = hin * B1in
      hB2[i] = hin * B2in
      PSI[i] = 0.

  elif choix == 60:

    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]

      AA = np.sin(2 * np.pi * xcent - 2 * np.pi * ycent)

      uu = 2.
      bb = 5.
      h[i] = 100
      hu[i] = uu + AA
      hv[i] = uu + AA
      hB1[i] = bb + 2 * AA
      hB2[i] = bb + 2 * AA
      PSI[i] = 0.

  elif choix == 42:

    # Magneto-Rossby wave on a beta-plane (Zaqarashvili et al.). Background:
    # uniform depth H0, uniform zonal (toroidal) field B0 x-hat, no mean flow.
    # A single Fourier-mode vortical perturbation from the streamfunction
    # psi = eps*cos(k1 x + k2 y) seeds the wave, so (u,v)=(-d psi/dy, d psi/dx).
    # Reuses the generic init args:
    #   k1, k2 = perturbation wavevector (kx, ky); eps = amplitude; tol = B0.
    # Requires the solver Coriolis source (f0, beta) to be active, and periodic
    # boundaries for a clean single-mode dispersion measurement.
    H0 = 1.0
    for i in range(nbelements):
      xcent = center[i][0]
      ycent = center[i][1]
      phase = k1 * xcent + k2 * ycent
      uu = eps * k2 * np.sin(phase)    # -d psi/dy
      vv = -eps * k1 * np.sin(phase)   #  d psi/dx
      h[i] = H0
      hu[i] = H0 * uu
      hv[i] = H0 * vv
      hB1[i] = H0 * tol                # tol := B0 (background zonal field)
      hB2[i] = 0.
      PSI[i] = 0.
      Z[i] = 0.


def _Total_Energy(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hB1_c: 'float[:]', hB2_c: 'float[:]',
                  Z_c: 'float[:]', grav: 'float', volumec: 'float[:]'):
  nbelement = len(h_c)
  num_t = 0.
  num_c = 0.
  num_m = 0.
  num_p = 0.

  for i in range(nbelement):
    hc = h_c[i]
    uc = hu_c[i] / h_c[i]
    vc = hv_c[i] / h_c[i]
    B1c = hB1_c[i] / h_c[i]
    B2c = hB2_c[i] / h_c[i]
    bc = Z_c[i]
    Ec = 0.5 * hc * (uc ** 2 + vc ** 2)              # Kinetic energy
    Em = 0.5 * hc * (B1c ** 2 + B2c ** 2)            # Magnetic energy
    Ep = 0.5 * grav * hc ** 2 + grav * bc * hc       # Potential energy
    Et = Ec + Em + Ep                                # Total energy
    num_t += volumec[i] * Et
    num_c += volumec[i] * Ec
    num_m += volumec[i] * Em
    num_p += volumec[i] * Ep

  return num_t, num_c, num_m, num_p


def _exact_solution_SWMHD(h_e: 'float[:]', hu_e: 'float[:]', hv_e: 'float[:]', hB1_e: 'float[:]', hB2_e: 'float[:]',
                          center: 'float[:,:]', time: 'float', grav: 'float'):
  # Exact solution of the smooth vortex (init choix=40) advected at velocity
  # (1,1), in CONSERVED variables (matches legacy Test67 error metric).
  nbelements = len(center)
  umax = 0.2
  Bmax = 0.1
  hmax = 1.0
  for i in range(nbelements):
    xcent = center[i][0]
    ycent = center[i][1]
    rcent = np.sqrt((xcent - time) ** 2 + (ycent - time) ** 2)
    ee = np.exp(1 - rcent ** 2)
    e1 = np.exp(0.5 * (1 - rcent ** 2))
    hin = hmax - (1. / (2.0 * grav)) * (umax ** 2 - Bmax ** 2) * ee
    h_e[i] = hin
    hu_e[i] = (1.0 - umax * e1 * (ycent - time)) * hin
    hv_e[i] = (1.0 + umax * e1 * (xcent - time)) * hin
    hB1_e[i] = (-Bmax * e1 * (ycent - time)) * hin
    hB2_e[i] = (Bmax * e1 * (xcent - time)) * hin


initialisation_SWMHD = FunObj(_initialisation_SWMHD)
Total_Energy = FunObj(_Total_Energy)
exact_solution_SWMHD = FunObj(_exact_solution_SWMHD)
