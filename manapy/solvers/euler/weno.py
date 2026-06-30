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

    # per cell, lists indexed by stencil (0 = central, 1.. = directional):
    self._stencils = []     # list of (list of stencil arrays)
    self._pinv = []         # list of (list of (K,M) pseudo-inverses)
    self._OI = []           # list of (list of K,K oscillation matrices)
    self._lam = []          # list of (linear weights per stencil)
    self._M0 = []           # cell-i basis averages (shared by all stencils)

    # cache cell -> triangles (fan from first vertex)
    cell_tris = []
    for i in range(self.nbcells):
      nv = nodeid[i][-1]
      vs = [verts[nodeid[i][j]] for j in range(nv)]
      tris = [(vs[0], vs[k], vs[k + 1]) for k in range(1, nv - 1)]
      cell_tris.append(tris)
    self._cell_tris = cell_tris

    # two-ring node-neighbour pool, used to draw central + directional stencils
    cn = [set(int(cellnid[i][j]) for j in range(cellnid[i][-1])) for i in range(self.nbcells)]
    m_central = int(np.ceil(1.8 * self.K))
    m_dir = int(np.ceil(1.5 * self.K))

    for i in range(self.nbcells):
      xi, yi = self.center[i]
      hi = self.h[i]
      M0 = np.array([self._cell_avg_monomial(i, xi, yi, hi, a, b) for (a, b) in self.exps])
      self._M0.append(M0)

      pool = set(cn[i])
      for m in list(cn[i]):
        pool |= cn[m]
      pool.discard(i)
      pool = np.array(sorted(pool), dtype=np.int32)
      d = self.center[pool] - self.center[i]
      dist = np.hypot(d[:, 0], d[:, 1])

      stencils = [self._select_central(pool, dist, m_central)]
      lams = [self.lambda_central]
      for k in range(self.ndir):
        ang = 2 * np.pi * k / self.ndir
        e = np.array([np.cos(ang), np.sin(ang)])
        stencils.append(self._select_directional(pool, d, dist, e, m_dir))
        lams.append(1.0)

      pinvs, ois = [], []
      for st in stencils:
        A = np.empty((len(st), self.K))
        for mloc, m in enumerate(st):
          for kk, (a, b) in enumerate(self.exps):
            A[mloc, kk] = self._cell_avg_monomial(m, xi, yi, hi, a, b) - M0[kk]
        pinvs.append(np.linalg.pinv(A))
        ois.append(self._oscillation_matrix(i))
      self._stencils.append(stencils)
      self._pinv.append(pinvs)
      self._OI.append(ois)
      self._lam.append(np.array(lams))

    # ---- pack the mesh-dependent data into padded arrays for the numba kernel ----
    ns = 1 + self.ndir
    max_m = max(len(st) for stl in self._stencils for st in stl)
    nc = self.nbcells
    self._st_idx = np.zeros((nc, ns, max_m), dtype=np.int32)
    self._st_cnt = np.zeros((nc, ns), dtype=np.int32)
    self._pinv_p = np.zeros((nc, ns, self.K, max_m))
    self._OI_p = np.zeros((nc, self.K, self.K))
    self._lam_arr = np.asarray(self._lam[0], dtype=float)   # same weights for all cells
    for i in range(nc):
      self._OI_p[i] = self._OI[i][0]
      for s in range(ns):
        st = self._stencils[i][s]
        m = len(st)
        self._st_cnt[i, s] = m
        self._st_idx[i, s, :m] = st
        self._pinv_p[i, s, :, :m] = self._pinv[i][s]
    global _weno_kernel_2d_compiled
    if _weno_kernel_2d_compiled is None:
      _weno_kernel_2d_compiled = compile(_weno_kernel_2d)
    self._kernel = _weno_kernel_2d_compiled

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
    """(1/|S_m|) * integral over cell m of ((x-xi)/hi)^a ((y-yi)/hi)^b."""
    acc = 0.0
    for (v0, v1, v2) in self._cell_tris[m]:
      acc += _tri_moment((v0[0] - xi, v0[1] - yi), (v1[0] - xi, v1[1] - yi),
                         (v2[0] - xi, v2[1] - yi), a, b)
    return acc / (self.vol[m] * hi ** (a + b))

  def _oscillation_matrix(self, i):
    """Per-cell OI_kq = sum over derivative multi-indices beta (1<=|beta|<=r) of
    the cell-averaged product of the beta-derivatives of the scaled basis
    monomials phi_k, phi_q (Eq 23). Symmetric positive semidefinite."""
    def ff(n, k):                                   # falling factorial n!/(n-k)!
      r = 1.0
      for j in range(k):
        r *= (n - j)
      return r
    xi, yi = self.center[i]
    hi = self.h[i]
    OI = np.zeros((self.K, self.K))
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
            OI[k, kq] += ck * cq * self._cell_avg_monomial(i, xi, yi, hi, A, B)
    return OI

  def reconstruct(self, U):
    """k-exact reconstruction on the **central** stencil only (linear; high order
    in smooth regions, oscillatory at discontinuities). Returns coeffs (nbcells, K)."""
    U = np.asarray(U)
    coeffs = np.zeros((self.nbcells, self.K))
    for i in range(self.nbcells):
      st = self._stencils[i][0]
      coeffs[i] = self._pinv[i][0] @ (U[st] - U[i])
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

  def smoothness(self, coeffs):
    """Smoothness indicator SI_i = a_i^T OI_i a_i for every cell (central stencil OI)."""
    si = np.empty(self.nbcells)
    for i in range(self.nbcells):
      a = coeffs[i]
      si[i] = float(a @ (self._OI[i][0] @ a))
    return si

  def evaluate(self, U, coeffs, i, x, y):
    """Evaluate the cell-i reconstruction polynomial at physical point (x, y)."""
    xi, yi = self.center[i]
    hi = self.h[i]
    val = U[i]
    for k, (a, b) in enumerate(self.exps):
      phi = ((x - xi) / hi) ** a * ((y - yi) / hi) ** b - self._M0[i][k]
      val += coeffs[i, k] * phi
    return val
