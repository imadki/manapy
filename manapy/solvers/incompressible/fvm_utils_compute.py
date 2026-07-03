#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-volume kernels for the collocated incompressible projection (icoFoam-like).

The method is face-flux consistent: the divergence, the pressure Laplacian and the
correction all share the SAME two-point face coefficient a_f = fv_coeff, so the
corrected face flux is divergence-free by construction (stable, no collocated
checkerboard blow-up). Momentum is transported by that divergence-free face flux.

Faces carry a name code: 0 interior, 10 partition (MPI halo neighbour), else a
physical boundary (wall/inlet). Partition faces use the exchanged halo cell value,
so the whole step is MPI-correct.
"""
from manapy.backends.compile_fun import compile


def _face_flux_2d(u_c: 'float[:]', v_c: 'float[:]', u_h: 'float[:]', v_h: 'float[:]',
                  uw: 'float[:]', vw: 'float[:]', normal: 'float[:,:]',
                  cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]', phi: 'float[:]'):
  # phi_f = u_face . S_f (area-scaled normal). Interior/partition: arithmetic average
  # (partition uses the halo neighbour); wall: the prescribed velocity.
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      uf = 0.5 * (u_c[iL] + u_c[iR]); vf = 0.5 * (v_c[iL] + v_c[iR])
    elif fname[f] == 10:
      h = halofid[f]
      uf = 0.5 * (u_c[iL] + u_h[h]); vf = 0.5 * (v_c[iL] + v_h[h])
    else:
      uf = uw[f]; vf = vw[f]
    phi[f] = uf * normal[f, 0] + vf * normal[f, 1]


def _mom_rhs_2d(u_c: 'float[:]', v_c: 'float[:]', u_h: 'float[:]', v_h: 'float[:]',
                phi: 'float[:]', af: 'float[:]', uw: 'float[:]', vw: 'float[:]',
                cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
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
    elif fname[f] == 10:                               # partition: owner side only
      h = halofid[f]
      uu = u_c[iL] if ph > 0.0 else u_h[h]
      vv = v_c[iL] if ph > 0.0 else v_h[h]
      du[iL] += -ph * uu + nu * a * (u_h[h] - u_c[iL])
      dv[iL] += -ph * vv + nu * a * (v_h[h] - v_c[iL])
    else:                                              # wall: phi~0, diffusion to wall vel
      du[iL] += -ph * u_c[iL] + nu * a * (uw[f] - u_c[iL])
      dv[iL] += -ph * v_c[iL] + nu * a * (vw[f] - v_c[iL])
  for i in range(n):
    du[i] /= vol[i]; dv[i] /= vol[i]


def _gg_grad_2d(P_c: 'float[:]', P_h: 'float[:]', normal: 'float[:,:]',
                cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
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
    elif fname[f] == 10:
      pf = 0.5 * (P_c[iL] + P_h[halofid[f]])
      gx[iL] += pf * normal[f, 0]; gy[iL] += pf * normal[f, 1]
    else:
      pf = P_c[iL]
      gx[iL] += pf * normal[f, 0]; gy[iL] += pf * normal[f, 1]
  for i in range(n):
    gx[i] /= vol[i]; gy[i] /= vol[i]


# ---------------------------------------------------------------------------
# True (implicit-momentum) PISO kernels
#
# The momentum equation is assembled implicitly as a global sparse matrix
#   M u = b0 - (V/rho) grad(p)
# with, per cell P (volume V_P) and face f (owner L, neighbour R, area-normal
# S_f pointing L->R, mass flux F = phi_f, two-point diffusion coeff a_f = fv_coeff):
#   time      : V_P/dt                       -> diagonal
#   convection: first-order upwind of F      -> diag += max(F,0), off-diag(L,R) -= max(-F,0)
#   diffusion : nu*a_f*(u_R-u_P)             -> diag += nu*a_f,  off-diag(L,R) -= nu*a_f
# so the per-face neighbour transfer coefficient is  cN = max(-F,0) + nu*a_f
# (owner->neighbour uses max(F,0)+nu*a_f). Dirichlet-velocity boundary faces fold
# their known value into the source b0 (bsu/bsv). Partition faces (name==10)
# couple the owner row to the halo neighbour's GLOBAL column (halosext[h,0]).
#
# The PISO pressure step then uses the Rhie-Chow pseudo-velocity HbyA = H/a_P
# (H = b0 - sum_N a_N u_N), the face coefficient D_f = a_f * interp(V/(rho*a_P)),
# and the variable-coefficient Laplacian div(D grad p) = div(phiHbyA).


def _mom_assemble_2d(massflux: 'float[:]', af: 'float[:]', uw: 'float[:]', vw: 'float[:]',
                     cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                     loctoglob: 'int64[:]', halosext: 'int64[:,:]', vol: 'float[:]',
                     dt: 'float', rho_c: 'float[:]', muf: 'float[:]',
                     aP: 'float[:]', bsu: 'float[:]',
                     bsv: 'float[:]', row: 'int64[:]', col: 'int64[:]', data: 'float[:]'):
  # Assemble the (variable-density) momentum matrix M (same for u and v) as global
  # triplets. Time term rho_P V/dt, convection by the MASS flux F = massflux_f
  # (first-order upwind), diffusion mu_f a_f. The off-diagonals are written first;
  # the diagonal block (aP) is appended last. Returns the total entry count.
  # `massflux` is rhoPhi (consistent with the alpha transport, Rudman) two-phase, or
  # rho_f phi single-phase; single-phase reduces to the scalar path (rho_c=rho, muf=rho*nu).
  nc = len(vol); nfc = len(cellid)
  for i in range(nc):
    aP[i] = rho_c[i] * vol[i] / dt; bsu[i] = 0.0; bsv[i] = 0.0
  cmpt = 0
  for f in range(nfc):
    iL = cellid[f, 0]; F = massflux[f]; nua = muf[f] * af[f]
    fp = F if F > 0.0 else 0.0                          # max(F, 0)
    fm = -F if F < 0.0 else 0.0                         # max(-F, 0)
    if fname[f] == 0:
      iR = cellid[f, 1]
      aP[iL] += fp + nua
      aP[iR] += fm + nua
      gL = loctoglob[iL]; gR = loctoglob[iR]
      row[cmpt] = gL; col[cmpt] = gR; data[cmpt] = -(fm + nua); cmpt += 1
      row[cmpt] = gR; col[cmpt] = gL; data[cmpt] = -(fp + nua); cmpt += 1
    elif fname[f] == 10:
      aP[iL] += fp + nua
      row[cmpt] = loctoglob[iL]; col[cmpt] = halosext[halofid[f], 0]
      data[cmpt] = -(fm + nua); cmpt += 1
    else:                                               # Dirichlet-velocity wall
      aP[iL] += fp + nua
      cN = fm + nua
      bsu[iL] += cN * uw[f]; bsv[iL] += cN * vw[f]
  for i in range(nc):
    g = loctoglob[i]
    row[cmpt] = g; col[cmpt] = g; data[cmpt] = aP[i]; cmpt += 1
  return cmpt


def _hbya_2d(un: 'float[:]', vn: 'float[:]', uc: 'float[:]', vc: 'float[:]',
             uh: 'float[:]', vh: 'float[:]', massflux: 'float[:]', af: 'float[:]',
             rho_c: 'float[:]', muf: 'float[:]',
             aP: 'float[:]', bsu: 'float[:]', bsv: 'float[:]', cellid: 'int64[:,:]',
             halofid: 'int64[:]', fname: 'int64[:]', vol: 'float[:]', dt: 'float',
             gsu: 'float[:]', gsv: 'float[:]', Hu: 'float[:]', Hv: 'float[:]'):
  # HbyA = H/a_P with H = rho_P (V/dt) u^n + boundary source + body force (gsu/gsv)
  # + sum_N cN u_N (frozen mass flux `massflux` = rhoPhi). Boundary contributions are
  # already in bsu/bsv. Must use the SAME massflux as _mom_assemble (same a_P split).
  # rho_c here must be the TIME-LEVEL-n density (conservative ddt: rho^n u^n V/dt
  # against the rho^{n+1} V/dt diagonal assembled in aP), so a uniform velocity stays
  # an exact solution across a moving interface (Galilean/mass-momentum consistency).
  nc = len(vol); nfc = len(cellid)
  for i in range(nc):
    Hu[i] = rho_c[i] * (vol[i] / dt) * un[i] + bsu[i] + gsu[i]
    Hv[i] = rho_c[i] * (vol[i] / dt) * vn[i] + bsv[i] + gsv[i]
  for f in range(nfc):
    iL = cellid[f, 0]; F = massflux[f]; nua = muf[f] * af[f]
    fp = F if F > 0.0 else 0.0
    fm = -F if F < 0.0 else 0.0
    if fname[f] == 0:
      iR = cellid[f, 1]
      cLR = fm + nua; cRL = fp + nua
      Hu[iL] += cLR * uc[iR]; Hv[iL] += cLR * vc[iR]
      Hu[iR] += cRL * uc[iL]; Hv[iR] += cRL * vc[iL]
    elif fname[f] == 10:
      h = halofid[f]; cLR = fm + nua
      Hu[iL] += cLR * uh[h]; Hv[iL] += cLR * vh[h]
  for i in range(nc):
    Hu[i] /= aP[i]; Hv[i] /= aP[i]


def _mom_ho_corr_2d(uc: 'float[:]', vc: 'float[:]', uh: 'float[:]', vh: 'float[:]',
                    gux: 'float[:]', guy: 'float[:]', gvx: 'float[:]', gvy: 'float[:]',
                    guxh: 'float[:]', guyh: 'float[:]', gvxh: 'float[:]', gvyh: 'float[:]',
                    massflux: 'float[:]', ccx: 'float[:]', ccy: 'float[:]',
                    fcx: 'float[:]', fcy: 'float[:]', hcx: 'float[:]', hcy: 'float[:]',
                    cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                    su: 'float[:]', sv: 'float[:]'):
  # DEFERRED second-order convection correction (limited linear-upwind). The implicit
  # momentum matrix stays first-order upwind (diagonally dominant, unconditionally
  # stable); this kernel returns the explicit conservative source su/sv (ADDED to
  # bsu/bsv) carrying the flux difference F (u_f^HO - u_up) per face, with
  #   u_f^HO = u_up + grad(u)_up . (x_f - x_up)
  # CLIPPED between the two cell values (minmod-like bound -> TVD, no new extrema;
  # reduces to first order across the sharp interface, second order in smooth flow).
  # At rest (u=0) the correction is identically zero, so the hydrostatic well-balance
  # is untouched. Boundary faces keep the prescribed wall flux (no correction).
  nc = len(uc); nfc = len(cellid)
  for i in range(nc):
    su[i] = 0.0; sv[i] = 0.0
  for f in range(nfc):
    F = massflux[f]
    if F == 0.0:
      continue
    iL = cellid[f, 0]
    if fname[f] == 0:
      iR = cellid[f, 1]
      if F > 0.0:
        uu = uc[iL]; vv = vc[iL]; ud = uc[iR]; vd = vc[iR]
        dx = fcx[f] - ccx[iL]; dy = fcy[f] - ccy[iL]
        du = gux[iL] * dx + guy[iL] * dy
        dv = gvx[iL] * dx + gvy[iL] * dy
      else:
        uu = uc[iR]; vv = vc[iR]; ud = uc[iL]; vd = vc[iL]
        dx = fcx[f] - ccx[iR]; dy = fcy[f] - ccy[iR]
        du = gux[iR] * dx + guy[iR] * dy
        dv = gvx[iR] * dx + gvy[iR] * dy
    elif fname[f] == 10:
      h = halofid[f]
      if F > 0.0:
        uu = uc[iL]; vv = vc[iL]; ud = uh[h]; vd = vh[h]
        dx = fcx[f] - ccx[iL]; dy = fcy[f] - ccy[iL]
        du = gux[iL] * dx + guy[iL] * dy
        dv = gvx[iL] * dx + gvy[iL] * dy
      else:
        uu = uh[h]; vv = vh[h]; ud = uc[iL]; vd = vc[iL]
        dx = fcx[f] - hcx[h]; dy = fcy[f] - hcy[h]
        du = guxh[h] * dx + guyh[h] * dy
        dv = gvxh[h] * dx + gvyh[h] * dy
    else:
      continue
    # clip the face value between the two cell values (bounded, TVD-like)
    lo = uu if uu < ud else ud; hi = uu if uu > ud else ud
    uf = uu + du
    if uf < lo:
      uf = lo
    elif uf > hi:
      uf = hi
    du = uf - uu
    lo = vv if vv < vd else vd; hi = vv if vv > vd else vd
    vf = vv + dv
    if vf < lo:
      vf = lo
    elif vf > hi:
      vf = hi
    dv = vf - vv
    cu = F * du; cv = F * dv                            # extra flux L->R through f
    su[iL] -= cu; sv[iL] -= cv
    if fname[f] == 0:
      su[cellid[f, 1]] += cu; sv[cellid[f, 1]] += cv


def _plap_assemble_2d(Df: 'float[:]', cellid: 'int64[:,:]', halofid: 'int64[:]',
                      fname: 'int64[:]', loctoglob: 'int64[:]', halosext: 'int64[:,:]',
                      vol: 'float[:]', pin: 'int64', diag: 'float[:]',
                      row: 'int64[:]', col: 'int64[:]', data: 'float[:]'):
  # PURE-NEUMANN two-point pressure Laplacian as global triplets, with global row `pin`
  # replaced by the identity (interFoam's pRefCell). Same row convention as ls_fv
  # (diag -D_f/V, off +D_f/V) but NO boundary terms: the walls are exactly closed, so
  # the singular Neumann system is compatible (sum of the div rhs = sum of wall fluxes
  # = 0) and pinning one row yields div(phi)=0 in EVERY cell -- including the pinned
  # one -- to solver precision. (A whole-wall Dirichlet reference over-determines the
  # wall rows instead: O(1) divergence residual in the reference band, which unbounds
  # the alpha transport there and destroys phase volume through the clip.)
  # Entries in the pinned row/column are dropped (p_pin = 0), keeping the matrix
  # symmetric. The pinned diagonal is emitted in the cell loop.
  nc = len(vol); nfc = len(cellid)
  for i in range(nc):
    diag[i] = 0.0
  cmpt = 0
  for f in range(nfc):
    iL = cellid[f, 0]; d = Df[f]
    if fname[f] == 0:
      iR = cellid[f, 1]
      diag[iL] -= d; diag[iR] -= d
      gL = loctoglob[iL]; gR = loctoglob[iR]
      if gL != pin and gR != pin:
        row[cmpt] = gL; col[cmpt] = gR; data[cmpt] = d / vol[iL]; cmpt += 1
        row[cmpt] = gR; col[cmpt] = gL; data[cmpt] = d / vol[iR]; cmpt += 1
    elif fname[f] == 10:
      diag[iL] -= d
      gL = loctoglob[iL]; gR = halosext[halofid[f], 0]
      if gL != pin and gR != pin:
        row[cmpt] = gL; col[cmpt] = gR; data[cmpt] = d / vol[iL]; cmpt += 1
  for i in range(nc):
    g = loctoglob[i]
    row[cmpt] = g; col[cmpt] = g
    data[cmpt] = 1.0 if g == pin else diag[i] / vol[i]
    cmpt += 1
  return cmpt


def _dcoeff_2d(rAU: 'float[:]', rAUh: 'float[:]', af: 'float[:]', cellid: 'int64[:,:]',
               halofid: 'int64[:]', fname: 'int64[:]', D: 'float[:]'):
  # Face pressure-Laplacian coefficient D_f = a_f * interp(rAU), rAU = V/(rho a_P).
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      D[f] = af[f] * 0.5 * (rAU[iL] + rAU[cellid[f, 1]])
    elif fname[f] == 10:
      D[f] = af[f] * 0.5 * (rAU[iL] + rAUh[halofid[f]])
    else:
      D[f] = af[f] * rAU[iL]


def _corr_flux_2d(phiHbyA: 'float[:]', D: 'float[:]', P_c: 'float[:]', P_h: 'float[:]',
                  cellid: 'int64[:,:]', halofid: 'int64[:]', fname: 'int64[:]',
                  phi: 'float[:]'):
  # Rhie-Chow flux correction phi_f = phiHbyA_f - D_f (p_N - p_P); boundary faces
  # keep the prescribed flux (already in phiHbyA).
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      phi[f] = phiHbyA[f] - D[f] * (P_c[cellid[f, 1]] - P_c[iL])
    elif fname[f] == 10:
      phi[f] = phiHbyA[f] - D[f] * (P_h[halofid[f]] - P_c[iL])
    else:
      phi[f] = phiHbyA[f]


_compiled = {}


def get_kernels():
  if not _compiled:
    _compiled['face_flux'] = compile(_face_flux_2d)
    _compiled['mom_rhs'] = compile(_mom_rhs_2d)
    _compiled['gg_grad'] = compile(_gg_grad_2d)
  return _compiled['face_flux'], _compiled['mom_rhs'], _compiled['gg_grad']


def _face_avg_2d(c: 'float[:]', c_h: 'float[:]', cellid: 'int64[:,:]',
                 halofid: 'int64[:]', fname: 'int64[:]', cf: 'float[:]'):
  # Arithmetic face value of a cell field (owner value at a physical boundary); used
  # for the face density rho_f and viscosity mu_f in the two-phase momentum.
  nfc = len(cellid)
  for f in range(nfc):
    iL = cellid[f, 0]
    if fname[f] == 0:
      cf[f] = 0.5 * (c[iL] + c[cellid[f, 1]])
    elif fname[f] == 10:
      cf[f] = 0.5 * (c[iL] + c_h[halofid[f]])
    else:
      cf[f] = c[iL]


_piso = {}


def get_piso_kernels():
  """Compile the implicit-momentum PISO kernels (momentum matrix assembly, HbyA,
  face pressure coefficient, Rhie-Chow flux correction)."""
  if not _piso:
    _piso['mom_assemble'] = compile(_mom_assemble_2d)
    _piso['hbya'] = compile(_hbya_2d)
    _piso['dcoeff'] = compile(_dcoeff_2d)
    _piso['corr_flux'] = compile(_corr_flux_2d)
    _piso['face_avg'] = compile(_face_avg_2d)
    _piso['plap'] = compile(_plap_assemble_2d)
    _piso['ho_corr'] = compile(_mom_ho_corr_2d)
  return (_piso['mom_assemble'], _piso['hbya'], _piso['dcoeff'], _piso['corr_flux'],
          _piso['face_avg'], _piso['plap'], _piso['ho_corr'])
