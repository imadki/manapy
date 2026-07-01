#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the 3D unstructured WENO reconstruction (weno.py).

  * moment geometry: the zeroth cell moment equals the cell volume;
  * k-exactness: a quadratic is reproduced to machine precision on the
    well-conditioned bulk of the mesh, with a bounded (regularised) pseudo-inverse
    everywhere (no blow-up on near-degenerate sliver/boundary stencils);
  * non-oscillation: the WENO weighting removes the Gibbs overshoot that the plain
    linear (central) reconstruction shows at a discontinuity.
"""
import os
import numpy as np
import pytest

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.weno import WenoReconstruction

MESH3D = os.path.join(os.path.dirname(__file__), "..", "meshes", "hybrid3d.msh")


@pytest.fixture(scope="module")
def weno():
  dom = Domain.create_domain(MESH3D, 3, Partitioning.Par_Nodal, recreate=True)
  return WenoReconstruction(dom, order=2)


def _cell_averages(dom, f):
  """Exact cell averages of f(p) via an independent 4-point (degree-2 exact)
  tetra quadrature over a centroid-apex face-triangulation of each cell."""
  V = np.asarray(dom.nodes.vertex)
  cf = np.asarray(dom.cells.faceid); fn = np.asarray(dom.faces.nodeid)
  cc = np.asarray(dom.cells.center); vol = np.asarray(dom.cells.volume)
  a4, b4 = 0.5854101966249685, 0.1381966011250105
  U = np.zeros(dom.nbcells)
  for i in range(dom.nbcells):
    integ = 0.0; c = cc[i]
    for jf in range(cf[i, -1]):
      fa = cf[i, jf]; nv = fn[fa, -1]; n0 = fn[fa, 0]
      for k in range(1, nv - 1):
        P = np.array([c, V[n0], V[fn[fa, k]], V[fn[fa, k + 1]]])
        Vt = abs(np.dot(P[1] - P[0], np.cross(P[2] - P[0], P[3] - P[0]))) / 6.0
        s = P.sum(0)
        qp = np.array([a4 * P[j] + b4 * (s - P[j]) for j in range(4)])
        integ += Vt * f(qp).mean()
    U[i] = integ / vol[i]
  return U


def test_weno3d_moment_is_volume(weno):
  vol = np.asarray(weno.domain.cells.volume)
  assert np.max(np.abs(weno._cm[:, 0] - vol) / vol) < 1e-10


def test_weno3d_pseudo_inverse_bounded(weno):
  pmax = max(np.max(np.abs(weno._pinv_p[i, 0, :, :weno._st_cnt[i, 0]]))
             for i in range(weno.nbcells))
  assert pmax < 1e3                                 # SVD truncation prevents blow-up


def test_weno3d_k_exact_quadratic(weno):
  dom = weno.domain
  cf = np.array([0.7, 1.3, -0.9, 0.5, 0.6, -0.4, 0.3, -0.5, 0.2, 0.15])

  def q(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return (cf[0] + cf[1] * x + cf[2] * y + cf[3] * z + cf[4] * x * x + cf[5] * y * y
            + cf[6] * z * z + cf[7] * x * y + cf[8] * x * z + cf[9] * y * z)

  U = _cell_averages(dom, q)
  co = weno.reconstruct(U)
  fc = np.asarray(dom.faces.center); fname = np.asarray(dom.faces.name)
  cid = np.asarray(dom.faces.cellid)
  errs = []
  for fi in range(len(fname)):
    if fname[fi] != 0 or weno._st_cnt[cid[fi, 0], 0] < weno.K:
      continue
    il = cid[fi, 0]
    errs.append(abs(weno.evaluate(U, co, il, fc[fi, 0], fc[fi, 1], fc[fi, 2]) - q(fc[fi])))
  errs = np.array(errs)
  # k-exact on the well-conditioned bulk (median), stable everywhere (bounded max)
  assert np.median(errs) < 1e-10
  assert np.percentile(errs, 90) < 1e-8
  assert np.max(errs) < 1.0                          # degenerate cells stay bounded


def test_weno3d_euler_sod_stable_nonoscillatory():
  """3D WENO-Euler coupling on a Sod shock tube: the SSP-RK3 run stays positive
  (rho, P > 0) and essentially non-oscillatory (rho within the initial [0.125, 1]
  bounds up to a tiny WENO overshoot)."""
  from manapy.core.Variable import Variable
  from manapy.solvers.euler.weno_euler import WenoEulerSolver
  dom = Domain.create_domain(MESH3D, 3, Partitioning.Par_Nodal, recreate=True)
  xc = np.asarray(dom.cells.center)[:, 0]
  gamma = 1.4
  xm = 0.5 * (xc.min() + xc.max())
  rho, rhou, rhov, rhow, rhoE = (Variable(domain=dom) for _ in range(5))
  left = xc < xm
  rho.cell[:] = np.where(left, 1.0, 0.125)
  rhoE.cell[:] = np.where(left, 1.0, 0.1) / (gamma - 1)
  bc = {k: "outflow" for k in ("in", "out", "upper", "bottom", "front", "back")}
  W = WenoEulerSolver(dom, rho.cell, rhou.cell, rhov.cell, rhoE.cell, rhow=rhow.cell,
                      gamma=gamma, cfl=0.3, bc=bc)
  t = 0.0
  while t < 0.1:
    dt = W.stepper()
    if t + dt > 0.1:
      dt = 0.1 - t
    W.step(dt); t += dt
  P = (gamma - 1) * (rhoE.cell - 0.5 * (rhou.cell ** 2 + rhov.cell ** 2 + rhow.cell ** 2) / rho.cell)
  assert np.all(rho.cell > 0) and np.all(P > 0)
  assert rho.cell.min() > 0.125 - 0.02 and rho.cell.max() < 1.0 + 0.02


def test_weno3d_non_oscillatory(weno):
  dom = weno.domain
  xc = np.asarray(dom.cells.center)[:, 0]
  xm = 0.5 * (xc.min() + xc.max())
  Ustep = np.where(xc < xm, 1.0, 0.0)
  cw = weno.weno_reconstruct(Ustep)
  cl = weno.reconstruct(Ustep)
  fc = np.asarray(dom.faces.center); fname = np.asarray(dom.faces.name)
  cid = np.asarray(dom.faces.cellid)
  ov_w = ov_l = 0.0
  for fi in range(len(fname)):
    if fname[fi] != 0:
      continue
    il = cid[fi, 0]
    vw = weno.evaluate(Ustep, cw, il, fc[fi, 0], fc[fi, 1], fc[fi, 2])
    vl = weno.evaluate(Ustep, cl, il, fc[fi, 0], fc[fi, 1], fc[fi, 2])
    ov_w = max(ov_w, vw - 1.0, -vw)
    ov_l = max(ov_l, vl - 1.0, -vl)
  assert ov_w < 1e-6                                 # WENO: essentially non-oscillatory
  assert ov_l > 0.1                                  # plain linear: clear Gibbs overshoot
