#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-volume kernels for the collocated incompressible projection (icoFoam-like).

The method is face-flux consistent: the divergence, the pressure Laplacian and the
correction all share the SAME two-point face coefficient a_f = area/dist, so the
corrected face flux is divergence-free by construction (stable, no collocated
checkerboard blow-up). Momentum is transported by that divergence-free face flux.
"""
from manapy.backends.compile_fun import compile


def _face_flux_2d(u_c: 'float[:]', v_c: 'float[:]', uw: 'float[:]', vw: 'float[:]',
                  normal: 'float[:,:]', cellid: 'int64[:,:]', fname: 'int64[:]', phi: 'float[:]'):
  # phi_f = u_face . S_f  (area-scaled normal). Interior: arithmetic face average;
  # boundary: the prescribed wall velocity (no through-flow at a closed wall).
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      uf = 0.5 * (u_c[iL] + u_c[iR]); vf = 0.5 * (v_c[iL] + v_c[iR])
    else:
      uf = uw[f]; vf = vw[f]
    phi[f] = uf * normal[f, 0] + vf * normal[f, 1]


def _mom_rhs_2d(u_c: 'float[:]', v_c: 'float[:]', phi: 'float[:]', af: 'float[:]',
                uw: 'float[:]', vw: 'float[:]', cellid: 'int64[:,:]', fname: 'int64[:]',
                vol: 'float[:]', nu: 'float', du: 'float[:]', dv: 'float[:]'):
  # d(u)/dt = (-conv + nu*diff)/vol. Convection uses the divergence-free face flux phi
  # (first-order upwind); diffusion is the two-point face gradient nu*a_f*(u_N-u_P).
  n = len(vol)
  for i in range(n):
    du[i] = 0.0; dv[i] = 0.0
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]; ph = phi[f]; a = af[f]
    if fname[f] == 0:
      iR = cellid[f, 1]
      uu = u_c[iL] if ph > 0.0 else u_c[iR]
      vv = v_c[iL] if ph > 0.0 else v_c[iR]
      fu = -ph * uu + nu * a * (u_c[iR] - u_c[iL])
      fv = -ph * vv + nu * a * (v_c[iR] - v_c[iL])
      du[iL] += fu; dv[iL] += fv
      du[iR] -= fu; dv[iR] -= fv
    else:                                              # wall: phi~0, diffusion to wall vel
      du[iL] += -ph * u_c[iL] + nu * a * (uw[f] - u_c[iL])
      dv[iL] += -ph * v_c[iL] + nu * a * (vw[f] - v_c[iL])
  for i in range(n):
    du[i] /= vol[i]; dv[i] /= vol[i]


def _gg_grad_2d(P_c: 'float[:]', normal: 'float[:,:]', cellid: 'int64[:,:]', fname: 'int64[:]',
                vol: 'float[:]', gx: 'float[:]', gy: 'float[:]'):
  # Green-Gauss cell gradient of P (for the collocated cell-velocity correction).
  n = len(vol)
  for i in range(n):
    gx[i] = 0.0; gy[i] = 0.0
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]; pf = 0.5 * (P_c[iL] + P_c[iR])
      gx[iL] += pf * normal[f, 0]; gy[iL] += pf * normal[f, 1]
      gx[iR] -= pf * normal[f, 0]; gy[iR] -= pf * normal[f, 1]
    else:
      pf = P_c[iL]
      gx[iL] += pf * normal[f, 0]; gy[iL] += pf * normal[f, 1]
  for i in range(n):
    gx[i] /= vol[i]; gy[i] /= vol[i]


_compiled = {}


def get_kernels():
  if not _compiled:
    _compiled['face_flux'] = compile(_face_flux_2d)
    _compiled['mom_rhs'] = compile(_mom_rhs_2d)
    _compiled['gg_grad'] = compile(_gg_grad_2d)
  return _compiled['face_flux'], _compiled['mom_rhs'], _compiled['gg_grad']
