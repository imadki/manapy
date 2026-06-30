#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-order WENO finite-volume reconstruction on unstructured meshes.

Implementation of Tsoutsanis, "Stencil selection algorithms for WENO schemes on
unstructured meshes", J. Comput. Phys. 475 (2023) 108840 (weno.pdf in this dir).

What is implemented (validated standalone):
  * k-exact least-squares reconstruction on the central (vertex-based) stencil:
    a polynomial p_i of order r built from the surrounding cell averages, with the
    zero-cell-average basis constraint (Eq 12) enforced through *exact* cell
    moments -> genuine k-exactness (degree <= r recovered exactly). The
    geometry-only pseudo-inverse A^+ (Eqs 8-16) is precomputed once per cell.
  * directional (sectoral) stencils drawn from the two-ring node-neighbour pool,
    one per direction, each with its own precomputed pseudo-inverse.
  * oscillation matrix OI (Eq 23) and smoothness indicator SI = a^T OI a (Eq 22),
    precomputed per cell; small for smooth data, large where data oscillate.
  * non-linear WENO weighting (Eqs 17-20): w_s = lam_s/(eps+SI_s)^power, with a
    large central linear weight. `weno_reconstruct` stays k-exact in smooth
    regions and essentially non-oscillatory at discontinuities (no Gibbs overshoot).

Everything that scales with the mesh is compiled with numba: the one-time build
(cell central moments, two-ring stencil selection, least-squares matrices,
SVD pseudo-inverses, oscillation matrices) runs in dedicated kernels, and so does
the per-step reconstruction/flux. Only the tiny mesh-independent "recipes" (the
binomial-shift and oscillation-matrix term lists, O(basis^2)) are built in Python.

Remaining: evaluate the WENO polynomial at edge Gauss points and wire it into the
Euler flux (replacing the MUSCL order-2 reconstruction); Shu-Osher validation.
"""
import numpy as np
from manapy.backends.compile_fun import compile


def _weno_kernel_2d(U: 'float64[:]', coeffs: 'float64[:,:]', st_idx: 'int32[:,:,:]',
                    st_cnt: 'int32[:,:]', pinv: 'float64[:,:,:,:]', OI: 'float64[:,:,:]',
                    lam: 'float64[:]', eps: 'float64', power: 'float64'):
    # Per-step WENO reconstruction hot loop (compiled). All geometry-dependent
    # arrays (stencil indices, padded pseudo-inverses, oscillation matrices,
    # linear weights) are precomputed once on the mesh and passed in; this only
    # does the data-dependent matvecs, smoothness quadratic forms and weighting.
    ncells = coeffs.shape[0]
    K = coeffs.shape[1]
    ns = lam.shape[0]
    a_s = np.zeros((ns, K))
    wbar = np.zeros(ns)
    for i in range(ncells):
        ui = U[i]
        for s in range(ns):
            cnt = st_cnt[i, s]
            for k in range(K):
                acc = 0.0
                for j in range(cnt):
                    acc += pinv[i, s, k, j] * (U[st_idx[i, s, j]] - ui)
                a_s[s, k] = acc
            si = 0.0
            for k in range(K):
                for q in range(K):
                    si += a_s[s, k] * OI[i, k, q] * a_s[s, q]
            wbar[s] = lam[s] / (eps + si) ** power
        wsum = 0.0
        for s in range(ns):
            wsum += wbar[s]
        for k in range(K):
            v = 0.0
            for s in range(ns):
                v += wbar[s] * a_s[s, k]
            coeffs[i, k] = v / wsum


_weno_kernel_2d_compiled = None


def _weno_build_2d(cm: 'float64[:,:]', cx: 'float64[:]', cy: 'float64[:]',
                   vol: 'float64[:]', h: 'float64[:]',
                   st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]',
                   amom: 'int32[:,:]', apdx: 'int32[:,:]', apdy: 'int32[:,:]',
                   acoef: 'float64[:,:]', acnt: 'int32[:]', aord: 'int32[:]', mono_cmidx: 'int32[:]',
                   oik: 'int32[:]', oiq: 'int32[:]', oimom: 'int32[:]', oicoef: 'float64[:]', oiord: 'int32[:]',
                   pinv_p: 'float64[:,:,:,:]', OI_p: 'float64[:,:,:]'):
    # Per-cell WENO build (compiled): for each stencil, form the least-squares
    # geometry matrix A from the precomputed central moments (binomial shift by the
    # stencil-local offset), then its pseudo-inverse via SVD (robust to ill-
    # conditioning); also assemble the oscillation matrix OI. No O(ncells) temporaries.
    nc = cm.shape[0]
    ns = st_cnt.shape[1]
    K = pinv_p.shape[2]
    max_m = pinv_p.shape[3]
    A = np.zeros((max_m, K))
    for i in range(nc):
        hi = h[i]; voli = vol[i]; xi = cx[i]; yi = cy[i]
        for s in range(ns):
            cnt = st_cnt[i, s]
            for j in range(cnt):
                m = st_idx[i, s, j]
                dx = cx[m] - xi; dy = cy[m] - yi
                volm = vol[m]
                for k in range(K):
                    integ = 0.0
                    for tt in range(acnt[k]):
                        integ += acoef[k, tt] * dx ** apdx[k, tt] * dy ** apdy[k, tt] * cm[m, amom[k, tt]]
                    avg_m = integ / (volm * hi ** aord[k])
                    m0k = cm[i, mono_cmidx[k]] / (voli * hi ** aord[k])
                    A[j, k] = avg_m - m0k
            # pseudo-inverse of A[:cnt, :K] via SVD: pinv = V diag(1/s) U^T
            U, sv, Vt = np.linalg.svd(A[:cnt, :])
            tol = 1e-14 * sv[0]                       # ~numpy pinv rcond; keeps k-exactness
            for k in range(K):
                for j in range(cnt):
                    acc = 0.0
                    for l in range(K):
                        if sv[l] > tol:
                            acc += Vt[l, k] * (1.0 / sv[l]) * U[j, l]
                    pinv_p[i, s, k, j] = acc
        # oscillation matrix OI[i] from cell-i central moments
        for t in range(oik.shape[0]):
            OI_p[i, oik[t], oiq[t]] += oicoef[t] * cm[i, oimom[t]] / (voli * hi ** oiord[t])


_weno_build_2d_compiled = None


def _weno_cm_2d(nodeid: 'int32[:,:]', vx: 'float64[:]', vy: 'float64[:]',
                cx: 'float64[:]', cy: 'float64[:]', cm: 'float64[:,:]'):
    # Central moments (order <= 2) of each cell about its own centre, by fan
    # triangulation. cm columns are [00,10,01,20,11,02].
    nc = nodeid.shape[0]
    last = nodeid.shape[1] - 1
    for i in range(nc):
        nv = nodeid[i, last]
        cmx = cx[i]; cmy = cy[i]
        x0 = vx[nodeid[i, 0]] - cmx; y0 = vy[nodeid[i, 0]] - cmy
        for k in range(1, nv - 1):
            x1 = vx[nodeid[i, k]] - cmx; y1 = vy[nodeid[i, k]] - cmy
            x2 = vx[nodeid[i, k + 1]] - cmx; y2 = vy[nodeid[i, k + 1]] - cmy
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            cm[i, 0] += area
            cm[i, 1] += area * (x0 + x1 + x2) / 3.0
            cm[i, 2] += area * (y0 + y1 + y2) / 3.0
            cm[i, 3] += area / 6.0 * (x0 * x0 + x1 * x1 + x2 * x2 + x0 * x1 + x1 * x2 + x2 * x0)
            cm[i, 4] += area / 12.0 * (2 * (x0 * y0 + x1 * y1 + x2 * y2)
                                       + x0 * y1 + x0 * y2 + x1 * y0 + x1 * y2 + x2 * y0 + x2 * y1)
            cm[i, 5] += area / 6.0 * (y0 * y0 + y1 * y1 + y2 * y2 + y0 * y1 + y1 * y2 + y2 * y0)


def _weno_select_2d(cellnid: 'int32[:,:]', cx: 'float64[:]', cy: 'float64[:]',
                    dirx: 'float64[:]', diry: 'float64[:]', m_central: 'int32', m_dir: 'int32',
                    st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]'):
    # Build the central + directional stencils from the two-ring node-neighbour pool.
    nc = cellnid.shape[0]
    last = cellnid.shape[1] - 1
    ndir = dirx.shape[0]
    poolmax = (last + 1) * (last + 1) + 4
    pool = np.empty(poolmax, dtype=np.int32)
    dist = np.empty(poolmax)
    score = np.empty(poolmax)
    for i in range(nc):
        npc = 0
        c1 = cellnid[i, last]
        for a in range(c1):
            m = cellnid[i, a]
            for b in range(-1, cellnid[m, last]):
                cand = m if b == -1 else cellnid[m, b]
                if cand == i:
                    continue
                dup = False
                for q in range(npc):
                    if pool[q] == cand:
                        dup = True; break
                if not dup and npc < poolmax:
                    pool[npc] = cand; npc += 1
        dmax = 1e-30
        for p in range(npc):
            dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]
            dist[p] = (dx * dx + dy * dy) ** 0.5
            if dist[p] > dmax:
                dmax = dist[p]
        # central stencil: the m_central closest cells
        order = np.argsort(dist[:npc])
        nce = m_central if m_central < npc else npc
        st_cnt[i, 0] = nce
        for j in range(nce):
            st_idx[i, 0, j] = pool[order[j]]
        # directional stencils: most aligned with each direction
        for k in range(ndir):
            ex = dirx[k]; ey = diry[k]
            for p in range(npc):
                dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]
                al = (dx * ex + dy * ey) / (dist[p] if dist[p] > 1e-30 else 1e-30)
                score[p] = -(al - 0.1 * dist[p] / dmax)     # ascending sort of -score
            od = np.argsort(score[:npc])
            nde = m_dir if m_dir < npc else npc
            st_cnt[i, 1 + k] = nde
            for j in range(nde):
                st_idx[i, 1 + k, j] = pool[od[j]]


_weno_cm_2d_compiled = None
_weno_select_2d_compiled = None


def _weno_advection_2d(rez: 'float64[:]', u_c: 'float64[:]', u_g: 'float64[:]',
                       coeffs: 'float64[:,:]', ea: 'int32[:]', eb: 'int32[:]', M0: 'float64[:,:]',
                       cx: 'float64[:]', cy: 'float64[:]', h: 'float64[:]',
                       fcx: 'float64[:]', fcy: 'float64[:]',
                       cellid: 'int32[:,:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                       name: 'uint32[:]', ax: 'float64', ay: 'float64'):
    # Linear-advection residual with a WENO-reconstructed upwind face value:
    # evaluate the WENO polynomial of the upwind cell at the face centre. The
    # high-order, non-oscillatory reconstruction is what gives WENO its quality.
    K = coeffs.shape[1]
    rez[:] = np.zeros(len(rez))
    nbface = len(cellid)
    for f in range(nbface):
        mes = mesure[f]
        nx = normal[f][0] / mes
        ny = normal[f][1] / mes
        un = ax * nx + ay * ny
        il = cellid[f][0]
        inner = name[f] == 0
        # pick the upwind cell and evaluate its reconstruction at the face centre
        if un >= 0.0:
            ic = il
        elif inner:
            ic = cellid[f][1]
        else:
            ic = -1                                 # boundary, use ghost below
        if ic >= 0:
            val = u_c[ic]
            xi = cx[ic]; yi = cy[ic]; hi = h[ic]
            for k in range(K):
                val += coeffs[ic, k] * (((fcx[f] - xi) / hi) ** ea[k]
                                        * ((fcy[f] - yi) / hi) ** eb[k] - M0[ic, k])
        else:
            val = u_g[f]
        flux = un * val * mes
        rez[il] -= flux
        if inner:
            rez[cellid[f][1]] += flux


_weno_advection_2d_compiled = None


# scaled monomial exponents for order r (2D), excluding the constant term
_EXPONENTS = {
    1: [(1, 0), (0, 1)],
    2: [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
    3: [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3)],
}


def _tri_moment(v0, v1, v2, a, b):
  """Exact integral of x^a y^b over a triangle (a+b <= 2), vertices v0,v1,v2."""
  x0, y0 = v0; x1, y1 = v1; x2, y2 = v2
  # absolute area: the per-vertex factors below are orientation-independent, only
  # the area magnitude enters the true geometric integral.
  area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
  if a == 0 and b == 0:
    return area
  if a == 1 and b == 0:
    return area * (x0 + x1 + x2) / 3.0
  if a == 0 and b == 1:
    return area * (y0 + y1 + y2) / 3.0
  if a == 2 and b == 0:
    return area / 6.0 * (x0 * x0 + x1 * x1 + x2 * x2 + x0 * x1 + x1 * x2 + x2 * x0)
  if a == 0 and b == 2:
    return area / 6.0 * (y0 * y0 + y1 * y1 + y2 * y2 + y0 * y1 + y1 * y2 + y2 * y0)
  if a == 1 and b == 1:
    return area / 12.0 * (2 * (x0 * y0 + x1 * y1 + x2 * y2)
                          + x0 * y1 + x0 * y2 + x1 * y0 + x1 * y2 + x2 * y0 + x2 * y1)
  raise ValueError("moment order > 2 not implemented")


class WenoReconstruction:

  def __init__(self, domain, order=2, ndir=4, lambda_central=1000.0, eps=1e-6, power=4.0):
    self.domain = domain
    self.order = int(order)
    if self.order not in _EXPONENTS:
      raise ValueError("order must be 1, 2 or 3")
    self.exps = _EXPONENTS[self.order]
    self.K = len(self.exps)

    self.ndir = int(ndir)                           # number of directional stencils
    self.lambda_central = float(lambda_central)
    self.eps = float(eps)
    self.power = float(power)

    cells = domain.cells
    self.nbcells = domain.nbcells
    self.center = np.asarray(cells.center)[:, :2]
    self.vol = np.asarray(cells.volume)
    self.h = np.sqrt(self.vol)                      # per-cell length scale
    cellnid = np.asarray(cells.cellnid)
    nodeid = np.asarray(cells.nodeid)
    verts = np.asarray(domain.nodes.vertex)[:, :2]

    nc = self.nbcells
    ns = 1 + self.ndir
    self._cm_idx = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (2, 0): 3, (1, 1): 4, (0, 2): 5}
    cx = np.ascontiguousarray(self.center[:, 0])
    cy = np.ascontiguousarray(self.center[:, 1])
    nodeid = np.ascontiguousarray(nodeid, dtype=np.int32)
    cellnid = np.ascontiguousarray(cellnid, dtype=np.int32)
    vx = np.ascontiguousarray(verts[:, 0])
    vy = np.ascontiguousarray(verts[:, 1])

    global _weno_cm_2d_compiled, _weno_select_2d_compiled
    if _weno_cm_2d_compiled is None:
      _weno_cm_2d_compiled = compile(_weno_cm_2d)
      _weno_select_2d_compiled = compile(_weno_select_2d)

    # central moments of every cell (numba)
    self._cm = np.zeros((nc, 6))                     # [00,10,01,20,11,02] about centre
    _weno_cm_2d_compiled(nodeid, vx, vy, cx, cy, self._cm)

    # central + directional stencils (numba)
    m_central = int(np.ceil(1.8 * self.K))
    m_dir = int(np.ceil(1.5 * self.K))
    max_m = max(m_central, m_dir)
    ang = 2 * np.pi * np.arange(self.ndir) / self.ndir
    dirx = np.ascontiguousarray(np.cos(ang)); diry = np.ascontiguousarray(np.sin(ang))
    self._st_idx = np.zeros((nc, ns, max_m), dtype=np.int32)
    self._st_cnt = np.zeros((nc, ns), dtype=np.int32)
    _weno_select_2d_compiled(cellnid, cx, cy, dirx, diry,
                             np.int32(m_central), np.int32(m_dir), self._st_idx, self._st_cnt)

    # ---- moment-shift / OI recipes (mesh-independent), then the numba build ----
    amom, apdx, apdy, acoef, acnt, aord, mono_cmidx = self._moment_recipe()
    oik, oiq, oimom, oicoef, oiord = self._oi_recipe()
    self._M0_p = np.empty((nc, self.K))
    for k, (a, b) in enumerate(self.exps):
      self._M0_p[:, k] = self._cm[:, self._cm_idx[(a, b)]] / (self.vol * self.h ** (a + b))
    self._pinv_p = np.zeros((nc, ns, self.K, max_m))
    self._OI_p = np.zeros((nc, self.K, self.K))
    global _weno_build_2d_compiled
    if _weno_build_2d_compiled is None:
      _weno_build_2d_compiled = compile(_weno_build_2d)
    _weno_build_2d_compiled(
        self._cm, np.ascontiguousarray(self.center[:, 0]), np.ascontiguousarray(self.center[:, 1]),
        self.vol, self.h, self._st_idx, self._st_cnt,
        amom, apdx, apdy, acoef, acnt, aord, mono_cmidx,
        oik, oiq, oimom, oicoef, oiord, self._pinv_p, self._OI_p)
    self._lam_arr = np.array([self.lambda_central] + [1.0] * self.ndir)

    global _weno_kernel_2d_compiled
    if _weno_kernel_2d_compiled is None:
      _weno_kernel_2d_compiled = compile(_weno_kernel_2d)
    self._kernel = _weno_kernel_2d_compiled

    # packed basis exponents / face geometry for the flux kernels
    self._ea = np.array([a for (a, b) in self.exps], dtype=np.int32)
    self._eb = np.array([b for (a, b) in self.exps], dtype=np.int32)
    fc = np.asarray(self.domain.faces.center)
    self._fcx = np.ascontiguousarray(fc[:, 0])
    self._fcy = np.ascontiguousarray(fc[:, 1])
    self._cx = np.ascontiguousarray(self.center[:, 0])
    self._cy = np.ascontiguousarray(self.center[:, 1])
    global _weno_advection_2d_compiled
    if _weno_advection_2d_compiled is None:
      _weno_advection_2d_compiled = compile(_weno_advection_2d)
    self._adv_kernel = _weno_advection_2d_compiled

  def _moment_recipe(self):
    """Per-monomial binomial-shift recipe: the integral of monomial (a,b) about a
    target is sum of terms coef * dx^pdx * dy^pdy * (central moment cm[mom])."""
    from math import comb
    K = self.K
    terms = []
    for (a, b) in self.exps:
      tk = []
      for p in range(a + 1):
        for q in range(b + 1):
          tk.append((self._cm_idx[(p, q)], a - p, b - q, float(comb(a, p) * comb(b, q))))
      terms.append(tk)
    mt = max(len(tk) for tk in terms)
    amom = np.zeros((K, mt), np.int32); apdx = np.zeros((K, mt), np.int32)
    apdy = np.zeros((K, mt), np.int32); acoef = np.zeros((K, mt))
    acnt = np.zeros(K, np.int32)
    aord = np.array([a + b for (a, b) in self.exps], np.int32)
    mono_cmidx = np.array([self._cm_idx[(a, b)] for (a, b) in self.exps], np.int32)
    for k, tk in enumerate(terms):
      acnt[k] = len(tk)
      for t, (mom, pdx, pdy, coef) in enumerate(tk):
        amom[k, t] = mom; apdx[k, t] = pdx; apdy[k, t] = pdy; acoef[k, t] = coef
    return amom, apdx, apdy, acoef, acnt, aord, mono_cmidx

  def _oi_recipe(self):
    """Mesh-independent oscillation-matrix recipe (Eq 23): list of (k, q, moment
    index, coefficient, moment order)."""
    def ff(n, k):
      r = 1.0
      for j in range(k):
        r *= (n - j)
      return r
    oik, oiq, oimom, oicoef, oiord = [], [], [], [], []
    for p in range(self.order + 1):
      for q in range(self.order + 1):
        if p + q < 1 or p + q > self.order:
          continue
        for k, (ak, bk) in enumerate(self.exps):
          if p > ak or q > bk:
            continue
          ck = ff(ak, p) * ff(bk, q)
          for kq, (aq, bq) in enumerate(self.exps):
            if p > aq or q > bq:
              continue
            cq = ff(aq, p) * ff(bq, q)
            A = (ak - p) + (aq - p)
            B = (bk - q) + (bq - q)
            oik.append(k); oiq.append(kq); oimom.append(self._cm_idx[(A, B)])
            oicoef.append(ck * cq); oiord.append(A + B)
    return (np.array(oik, np.int32), np.array(oiq, np.int32), np.array(oimom, np.int32),
            np.array(oicoef), np.array(oiord, np.int32))

  def reconstruct(self, U):
    """k-exact reconstruction on the **central** stencil only (linear; high order
    in smooth regions, oscillatory at discontinuities). Returns coeffs (nbcells, K)."""
    U = np.asarray(U)
    coeffs = np.zeros((self.nbcells, self.K))
    for i in range(self.nbcells):
      cnt = self._st_cnt[i, 0]
      st = self._st_idx[i, 0, :cnt]
      coeffs[i] = self._pinv_p[i, 0, :, :cnt] @ (U[st] - U[i])
    return coeffs

  def weno_reconstruct(self, U):
    """Non-linear WENO reconstruction (Eqs 17-23): blend the central and directional
    stencil polynomials with weights w_s = lam_s/(eps+SI_s)^power, normalised. In
    smooth regions the large central linear weight dominates (high order); near a
    discontinuity the stencils that cross it get a large SI and are suppressed, so
    the reconstruction stays essentially non-oscillatory. Returns coeffs (nbcells, K).

    Runs the compiled (numba) hot loop over the precomputed mesh-dependent arrays."""
    U = np.ascontiguousarray(U, dtype=float)
    coeffs = np.zeros((self.nbcells, self.K))
    self._kernel(U, coeffs, self._st_idx, self._st_cnt, self._pinv_p,
                 self._OI_p, self._lam_arr, self.eps, self.power)
    return coeffs

  def advect_residual(self, u_cell, u_ghost, coeffs, ax, ay):
    """WENO linear-advection residual d(u*vol)/dt for velocity (ax, ay):
    upwind flux with the WENO polynomial evaluated at face centres."""
    rez = np.zeros(self.nbcells)
    self._adv_kernel(rez, np.ascontiguousarray(u_cell, dtype=float),
                     np.ascontiguousarray(u_ghost, dtype=float), coeffs,
                     self._ea, self._eb, self._M0_p, self._cx, self._cy, self.h,
                     self._fcx, self._fcy, self.domain.faces.cellid,
                     self.domain.faces.normal, self.domain.faces.mesure,
                     np.asarray(self.domain.faces.name, dtype=np.uint32), float(ax), float(ay))
    return rez

  def smoothness(self, coeffs):
    """Smoothness indicator SI_i = a_i^T OI_i a_i for every cell."""
    return np.einsum('ik,ikq,iq->i', coeffs, self._OI_p, coeffs)

  def evaluate(self, U, coeffs, i, x, y):
    """Evaluate the cell-i reconstruction polynomial at physical point (x, y)."""
    xi, yi = self.center[i]
    hi = self.h[i]
    val = U[i]
    for k, (a, b) in enumerate(self.exps):
      phi = ((x - xi) / hi) ** a * ((y - yi) / hi) ** b - self._M0_p[i][k]
      val += coeffs[i, k] * phi
    return val
