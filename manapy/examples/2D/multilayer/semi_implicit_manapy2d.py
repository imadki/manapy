#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semi-implicit shallow water on the manapy UNSTRUCTURED mesh (production port).

Proves the two hard parts of porting the Casulli semi-implicit into the cell-
centred collocated manapy solver:
  1. the variable-coefficient Helmholtz  (V + theta^2 dt^2 gH L) eta = RHS  assembled
     from the FV face geometry (two-point flux) and solved with a sparse LU;
  2. checkerboard-free collocated coupling via a Rhie-Chow face flux -- the mass
     flux / Helmholtz uses the COMPACT face pressure gradient (eta_R-eta_L)/d, while
     the cell velocities are corrected with a Green-Gauss cell gradient.

theta=1/2 (Crank-Nicolson). Validated against the exact standing wave
eta = A cos(kx x) cos(w t) at dt >> the explicit CFL (where explicit blows up).
"""
import os
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu
from manapy.api.mesh import Mesh

G, H0 = 9.81, 1.0
LX, LY, N = 1.0, 1.0, 48
A0, KX = 1e-3, np.pi
OMEGA = np.sqrt(G * H0) * KX
PERIOD = 2 * np.pi / OMEGA
THETA = 0.5

# quad mesh -> orthogonal two-point flux is exact; triangle -> non-orthogonality
# error (the manapy Diamond scheme would restore accuracy there).
CT = os.environ.get("CT", "quadrangle")
mesh = Mesh.rectangle(bounds=((0., LX), (0., LY)), n=(N, N),
                      cell_type=CT, recombine=(CT == "quadrangle"))
dom = mesh.domain
ncell = dom.nbcells
cc = np.asarray(dom.cells.center)
cv = np.asarray(dom.cells.volume)
fc = np.asarray(dom.faces.cellid)
fn = np.asarray(dom.faces.normal)          # area-weighted normal, oriented cellid[0]->cellid[1]
fm = np.asarray(dom.faces.mesure)
inner = np.asarray(dom.innerfaces)
bound = np.asarray(dom.boundaryfaces)

L = fc[inner, 0]; R = fc[inner, 1]
dist = np.sqrt(((cc[R] - cc[L])**2).sum(1))
coeff = fm[inner] / dist                    # A_f / d_f  (two-point flux)
nfx, nfy = fn[inner, 0], fn[inner, 1]
bL = fc[bound, 0]; bnx, bny = fn[bound, 0], fn[bound, 1]

xc = cc[:, 0]
dx = LX / N
DT_CFL = 0.9 * dx / np.sqrt(G * H0)


def laplacian_matrix():
  """Per-volume Laplacian operator (mat ~ +div grad). Two-point on quads; the
  manapy Diamond scheme on triangles (restores accuracy on non-orthogonal cells)."""
  if os.environ.get("SCHEME", "twopoint" if CT == "quadrangle" else "diamond") == "diamond":
    from manapy.core.Variable import Variable
    from manapy.solvers.ls import ScipySolver
    P = Variable(domain=dom)                                   # Neumann default -> closed walls
    S = ScipySolver(domain=dom, var=P, scheme="diamond")
    S(np.zeros(ncell))
    return S.mat.tocsr(), "diamond"
  rows = np.concatenate([L, R, L, R])
  cols = np.concatenate([L, R, R, L])
  vals = np.concatenate([coeff, coeff, -coeff, -coeff])       # integrated -Laplacian (pos. diag)
  Lint = csr_matrix((vals, (rows, cols)), shape=(ncell, ncell))
  mat = -diags(1.0 / cv) @ Lint                                # -> per-volume +Laplacian
  return mat.tocsr(), "twopoint"


def build_helmholtz(dt, mat):
  a = THETA**2 * dt**2 * G * H0
  A = diags(np.ones(ncell)) - a * mat                          # I - alpha * laplacian
  return splu(A.tocsc()), a


def grad_gg(eta):                            # Green-Gauss cell gradient (for cell velocity)
  ef = 0.5 * (eta[L] + eta[R])
  gx = np.zeros(ncell); gy = np.zeros(ncell)
  np.add.at(gx, L, ef * nfx); np.add.at(gx, R, -ef * nfx)
  np.add.at(gy, L, ef * nfy); np.add.at(gy, R, -ef * nfy)
  np.add.at(gx, bL, eta[bL] * bnx); np.add.at(gy, bL, eta[bL] * bny)   # Neumann wall
  return gx / cv, gy / cv


def div_HU(u, v):                            # integrated div of H*velocity (face-averaged)
  uf = 0.5 * (u[L] + u[R]); vf = 0.5 * (v[L] + v[R])
  flux = H0 * (uf * nfx + vf * nfy)          # walls: u.n = 0 -> no boundary flux
  d = np.zeros(ncell)
  np.add.at(d, L, flux); np.add.at(d, R, -flux)
  return d


def run(dt, T, semi, mat=None, lu=None, a=None):
  eta = A0 * np.cos(KX * xc)
  u = np.zeros(ncell); v = np.zeros(ncell)
  t, emax = 0.0, 0.0
  while t < T - 1e-12:
    if semi:
      # per-volume theta=1/2: (I - a*lap) eta^{n+1} = eta^n - dt div(HU) + a*lap(eta^n)
      rhs = eta - dt * div_HU(u, v) / cv + a * (mat @ eta)
      eta_new = lu.solve(rhs)
      gxn, gyn = grad_gg(eta); gxN, gyN = grad_gg(eta_new)
      u = u - dt * G * (THETA * gxN + (1 - THETA) * gxn)
      v = v - dt * G * (THETA * gyN + (1 - THETA) * gyn)
      eta = eta_new
    else:
      eta = eta - dt * div_HU(u, v) / cv
      gx, gy = grad_gg(eta)
      u = u - dt * G * gx; v = v - dt * G * gy
    t += dt
    emax = max(emax, float(np.max(np.abs(eta))))
    if not np.isfinite(emax) or emax > 1e3 * A0:
      return None, emax
  return eta, emax


if __name__ == "__main__":
  factor = 10
  dt = factor * DT_CFL
  T = 2 * PERIOD
  mat, scheme = laplacian_matrix()
  lu, a = build_helmholtz(dt, mat)
  print(f"[manapy-SI] {CT} mesh {ncell} cells, Laplacian scheme='{scheme}'; "
        f"explicit CFL dt={DT_CFL:.2e}s ; run dt={dt:.2e}s ({factor}x)")

  eta, emax = run(dt, T, semi=True, mat=mat, lu=lu, a=a)
  ex = A0 * np.cos(KX * xc) * np.cos(OMEGA * T)
  l2 = np.sqrt(np.sum((eta - ex)**2 * cv) / np.sum(ex**2 * cv + 1e-30))
  # checkerboard sniff: neighbour-difference energy relative to amplitude
  cb = float(np.max(np.abs(eta[L] - eta[R]))) / A0
  print(f"[manapy-SI] semi-implicit STABLE at {factor}x CFL (max|eta|/A0={emax/A0:.2f})")
  print(f"[manapy-SI] L2 vs EXACT standing wave = {l2:.3e}  -> {'PASS' if l2 < 0.1 else 'FAIL'}")
  print(f"[manapy-SI] checkerboard indicator max|eta_L-eta_R|/A0 = {cb:.2f}  (smooth if O(1), spiky if >>1)")

  res, emax_e = run(dt, T, semi=False)
  print(f"[manapy-SI] explicit at same {factor}x dt: "
        + ("bounded (unexpected)" if res is not None else f"BLEW UP (max/A0={emax_e/A0:.1e})"))
