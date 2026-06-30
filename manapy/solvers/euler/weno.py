#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-order WENO finite-volume reconstruction on unstructured meshes.

Implementation of Tsoutsanis, "Stencil selection algorithms for WENO schemes on
unstructured meshes", J. Comput. Phys. 475 (2023) 108840 (weno.pdf in this dir).

Milestone 1 -- the foundation: k-exact least-squares polynomial reconstruction on
the central (vertex-based) stencil. For each cell i a polynomial p_i of order r is
built from the surrounding cell averages so that its own average matches U_i and
it reproduces the neighbours' averages in a least-squares sense (Eqs 8-16). The
basis is a scaled, cell-centred monomial set with the zero-cell-average
constraint (Eq 12) enforced through *exact* cell moments, which gives genuine
k-exactness: a polynomial field of degree <= r is recovered exactly.

The geometry-only pseudo-inverse A^+ (Eq 15-16) is precomputed once per cell; the
per-step reconstruction is then a single matvec b -> a.

Later milestones: nonlinear WENO weighting of central + directional stencils
(smoothness indicators, Eqs 17-23), edge Gauss-quadrature flux, solver coupling.
"""
import numpy as np


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

  def __init__(self, domain, order=2):
    self.domain = domain
    self.order = int(order)
    if self.order not in _EXPONENTS:
      raise ValueError("order must be 1, 2 or 3")
    self.exps = _EXPONENTS[self.order]
    self.K = len(self.exps)

    cells = domain.cells
    self.nbcells = domain.nbcells
    self.center = np.asarray(cells.center)[:, :2]
    self.vol = np.asarray(cells.volume)
    self.h = np.sqrt(self.vol)                      # per-cell length scale
    cellnid = np.asarray(cells.cellnid)
    nodeid = np.asarray(cells.nodeid)
    verts = np.asarray(domain.nodes.vertex)[:, :2]

    # precompute, per cell, raw geometric moments mu[m][(a,b)] = integral over
    # cell m of (x-xi)^a (y-yi)^b ... done lazily inside the loop per target cell.
    self._coeff_arrays = None
    self._stencils = []
    self._pinv = []                                 # (K, M) pseudo-inverse per cell
    self._M0 = []                                   # cell-i average of each basis monomial

    # cache cell -> triangles (fan from first vertex)
    cell_tris = []
    for i in range(self.nbcells):
      nv = nodeid[i][-1]
      vs = [verts[nodeid[i][j]] for j in range(nv)]
      tris = [(vs[0], vs[k], vs[k + 1]) for k in range(1, nv - 1)]
      cell_tris.append(tris)
    self._cell_tris = cell_tris

    for i in range(self.nbcells):
      ncn = cellnid[i][-1]
      stencil = [int(cellnid[i][j]) for j in range(ncn)]
      self._stencils.append(np.array(stencil, dtype=np.int32))
      xi, yi = self.center[i]
      hi = self.h[i]

      # cell-i averages of the scaled monomials (for the zero-average constraint)
      M0 = np.array([self._cell_avg_monomial(i, xi, yi, hi, a, b) for (a, b) in self.exps])
      self._M0.append(M0)

      # least-squares geometry matrix A_mk = avg_m(psi_k) - avg_i(psi_k)
      A = np.empty((len(stencil), self.K))
      for mloc, m in enumerate(stencil):
        for k, (a, b) in enumerate(self.exps):
          A[mloc, k] = self._cell_avg_monomial(m, xi, yi, hi, a, b) - M0[k]
      # Moore-Penrose pseudo-inverse (Eq 15-16); geometry only, precomputed once
      self._pinv.append(np.linalg.pinv(A))

  def _cell_avg_monomial(self, m, xi, yi, hi, a, b):
    """(1/|S_m|) * integral over cell m of ((x-xi)/hi)^a ((y-yi)/hi)^b."""
    acc = 0.0
    for (v0, v1, v2) in self._cell_tris[m]:
      acc += _tri_moment((v0[0] - xi, v0[1] - yi), (v1[0] - xi, v1[1] - yi),
                         (v2[0] - xi, v2[1] - yi), a, b)
    return acc / (self.vol[m] * hi ** (a + b))

  def reconstruct(self, U):
    """Return the reconstruction coefficients a_k for every cell, shape (nbcells, K).

    The reconstructed polynomial in cell i is  p_i(x,y) = U_i + sum_k a_k phi_k,
    phi_k = ((x-xi)/hi)^ak ((y-yi)/hi)^bk - M0_k."""
    U = np.asarray(U)
    coeffs = np.zeros((self.nbcells, self.K))
    for i in range(self.nbcells):
      st = self._stencils[i]
      b = U[st] - U[i]
      coeffs[i] = self._pinv[i] @ b
    return coeffs

  def evaluate(self, U, coeffs, i, x, y):
    """Evaluate the cell-i reconstruction polynomial at physical point (x, y)."""
    xi, yi = self.center[i]
    hi = self.h[i]
    val = U[i]
    for k, (a, b) in enumerate(self.exps):
      phi = ((x - xi) / hi) ** a * ((y - yi) / hi) ** b - self._M0[i][k]
      val += coeffs[i, k] * phi
    return val
