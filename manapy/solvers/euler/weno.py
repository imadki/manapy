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

    # cache cell -> triangles (fan from first vertex)
    cell_tris = []
    for i in range(self.nbcells):
      nv = nodeid[i][-1]
      vs = [verts[nodeid[i][j]] for j in range(nv)]
      tris = [(vs[0], vs[k], vs[k + 1]) for k in range(1, nv - 1)]
      cell_tris.append(tris)
    self._cell_tris = cell_tris

    # Central moments of each cell about its OWN centre (order <= 2), computed once
    # (accurate -- small quantities, no cancellation). The average of a monomial of
    # cell m about any target centre (xi,yi) is then a binomial shift by the
    # *stencil-local* offset (centre_m - (xi,yi)), which is small -> still accurate,
    # and avoids the per-(target, cell, monomial) triangle loop (build hot spot).
    self._cm = np.zeros((self.nbcells, 6))          # [00,10,01,20,11,02] about centre
    _abs = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for m in range(self.nbcells):
      cmx, cmy = self.center[m]
      for (v0, v1, v2) in cell_tris[m]:
        sv = ((v0[0] - cmx, v0[1] - cmy), (v1[0] - cmx, v1[1] - cmy), (v2[0] - cmx, v2[1] - cmy))
        for c, (a, b) in enumerate(_abs):
          self._cm[m, c] += _tri_moment(sv[0], sv[1], sv[2], a, b)

    nc = self.nbcells
    ns = 1 + self.ndir
    self._cm_idx = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (2, 0): 3, (1, 1): 4, (0, 2): 5}

    # ---- stencil selection (per-cell pool logic; cheap, no moments) ----
    cn = [set(int(cellnid[i][j]) for j in range(cellnid[i][-1])) for i in range(nc)]
    m_central = int(np.ceil(1.8 * self.K))
    m_dir = int(np.ceil(1.5 * self.K))
    stencils_all = []
    for i in range(nc):
      pool = set(cn[i])
      for m in cn[i]:
        pool |= cn[m]
      pool.discard(i)
      pool = np.array(sorted(pool), dtype=np.int32)
      d = self.center[pool] - self.center[i]
      dist = np.hypot(d[:, 0], d[:, 1])
      sts = [self._select_central(pool, dist, m_central)]
      for k in range(self.ndir):
        ang = 2 * np.pi * k / self.ndir
        sts.append(self._select_directional(pool, d, dist,
                                             np.array([np.cos(ang), np.sin(ang)]), m_dir))
      stencils_all.append(sts)
    max_m = max(len(st) for sts in stencils_all for st in sts)

    self._st_idx = np.zeros((nc, ns, max_m), dtype=np.int32)
    self._st_cnt = np.zeros((nc, ns), dtype=np.int32)
    for i in range(nc):
      for s in range(ns):
        st = stencils_all[i][s]
        self._st_cnt[i, s] = len(st)
        self._st_idx[i, s, :len(st)] = st

    # ---- vectorised moments / A-matrix / pseudo-inverse (the build hot path) ----
    cm = self._cm
    ctr = self.center
    M = self._st_idx                                  # (nc, ns, max_m) cell ids
    dx = ctr[M, 0] - ctr[:, None, None, 0]
    dy = ctr[M, 1] - ctr[:, None, None, 1]
    u = cm[M]                                          # (nc, ns, max_m, 6)
    u00, u10, u01, u20, u11, u02 = (u[..., j] for j in range(6))
    hI = self.h[:, None, None]
    volM = self.vol[M]
    # cell-i basis averages M0 (m = i, zero shift)
    self._M0_p = np.empty((nc, self.K))
    A = np.empty((nc, ns, max_m, self.K))
    for k, (a, b) in enumerate(self.exps):
      if (a, b) == (1, 0): integ = u10 + dx * u00
      elif (a, b) == (0, 1): integ = u01 + dy * u00
      elif (a, b) == (2, 0): integ = u20 + 2 * dx * u10 + dx * dx * u00
      elif (a, b) == (1, 1): integ = u11 + dx * u01 + dy * u10 + dx * dy * u00
      elif (a, b) == (0, 2): integ = u02 + 2 * dy * u01 + dy * dy * u00
      avg_m = integ / (volM * hI ** (a + b))
      M0k = cm[:, self._cm_idx[(a, b)]] / (self.vol * self.h ** (a + b))
      self._M0_p[:, k] = M0k
      A[:, :, :, k] = avg_m - M0k[:, None, None]
    valid = np.arange(max_m)[None, None, :] < self._st_cnt[:, :, None]
    A *= valid[..., None]                              # zero the padded rows
    pinv = np.linalg.pinv(A.reshape(nc * ns, max_m, self.K))   # batched (nc*ns, K, max_m)
    self._pinv_p = np.ascontiguousarray(pinv.reshape(nc, ns, self.K, max_m))

    # ---- vectorised oscillation matrices OI (one per cell) ----
    self._OI_p = self._oscillation_matrices()
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

  @staticmethod
  def _select_central(pool, dist, m):
    return pool[np.argsort(dist)[:min(m, len(pool))]]

  @staticmethod
  def _select_directional(pool, d, dist, e, m):
    # cells most aligned with direction e (one-sided), nearest first among them
    align = (d[:, 0] * e[0] + d[:, 1] * e[1]) / np.maximum(dist, 1e-30)
    score = align - 0.1 * dist / max(dist.max(), 1e-30)
    return pool[np.argsort(-score)[:min(m, len(pool))]]

  def _cell_avg_monomial(self, m, xi, yi, hi, a, b):
    """(1/|S_m|) * integral over cell m of ((x-xi)/hi)^a ((y-yi)/hi)^b, by binomial
    shift of cell m's precomputed central moments by the stencil-local offset
    (centre_m - (xi,yi)). Small shift -> accurate; no triangle loop."""
    u00, u10, u01, u20, u11, u02 = self._cm[m]
    dx = self.center[m, 0] - xi
    dy = self.center[m, 1] - yi
    if a == 0 and b == 0:
      integ = u00
    elif a == 1 and b == 0:
      integ = u10 + dx * u00
    elif a == 0 and b == 1:
      integ = u01 + dy * u00
    elif a == 2 and b == 0:
      integ = u20 + 2 * dx * u10 + dx * dx * u00
    elif a == 0 and b == 2:
      integ = u02 + 2 * dy * u01 + dy * dy * u00
    elif a == 1 and b == 1:
      integ = u11 + dx * u01 + dy * u10 + dx * dy * u00
    else:
      raise ValueError("moment order > 2 not implemented")
    return integ / (self.vol[m] * hi ** (a + b))

  def _oscillation_matrices(self):
    """Vectorised oscillation matrices OI_kq for all cells (Eq 23):
    sum over derivative multi-indices beta (1<=|beta|<=r) of the cell-averaged
    product of the beta-derivatives of the scaled basis monomials. The (p,q,k,q)
    recipe is mesh-independent; each term is one array op over all cells."""
    nc, K = self.nbcells, self.K
    orders = np.array([0, 1, 1, 2, 2, 2])
    avgi = self._cm / (self.vol[:, None] * self.h[:, None] ** orders[None, :])  # (nc, 6)

    def ff(n, k):
      r = 1.0
      for j in range(k):
        r *= (n - j)
      return r

    OI = np.zeros((nc, K, K))
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
            OI[:, k, kq] += ck * cq * avgi[:, self._cm_idx[(A, B)]]
    return OI

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
