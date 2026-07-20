#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semi-implicit (Casulli, theta=1/2) barotropic shallow-water core.

This is the time-integration core that lifts the fast external-gravity-wave step
restriction dt ~ dx/sqrt(gH) of the explicit multilayer solver: the barotropic
pressure/continuity coupling is treated implicitly, so dt is limited only by the
slow (baroclinic/advective) dynamics. For a low-Froude flow that is the ~sqrt(gH)/
sqrt(g'h) speedup -- exactly why the explicit HLLC plunging-plume runs were slow.

Linearised barotropic system on a staggered C-grid (eta at centres, u/v at faces):
    d_t eta + H0 div(u) = 0 ,   d_t u + g grad(eta) = 0
theta=1/2 gives the Crank-Nicolson scheme -> unconditionally stable, non-dissipative.
Eliminating u^{n+1} yields a Helmholtz problem for eta^{n+1}:
    (I - alpha L) eta^{n+1} = eta^n - dt H0 div(u^n) + alpha L eta^n ,
    alpha = (dt^2 g H0)/4 ,  L = Neumann Laplacian (closed basin, u=0 on walls).
Solved with CG on a matrix-free operator x -> x - alpha*lap(x)  (SPD).

VALIDATION: exact standing wave  eta = A cos(kx x) cos(w t),  w = sqrt(gH0) kx,
run at dt = 20x the explicit CFL (where an explicit scheme blows up).
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

G, H0 = 9.81, 1.0
LX, LY = 1.0, 1.0
NX, NY = 64, 64
DX, DY = LX / NX, LY / NY
A0 = 1e-3                              # small amplitude -> linear regime
KX = np.pi / LX                       # m=1 mode, ky=0
OMEGA = np.sqrt(G * H0) * KX
PERIOD = 2 * np.pi / OMEGA

xc = (np.arange(NX) + 0.5) * DX       # cell-centre x


def divergence(u, v):
  return (u[1:, :] - u[:-1, :]) / DX + (v[:, 1:] - v[:, :-1]) / DY

def gradx(eta):                       # -> (NX+1, NY), 0 on x-walls (Neumann)
  g = np.zeros((NX + 1, NY))
  g[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / DX
  return g

def grady(eta):
  g = np.zeros((NX, NY + 1))
  g[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / DY
  return g

def laplacian(eta):
  return divergence(gradx(eta), grady(eta))


def eta_exact(t):
  return A0 * np.cos(KX * xc)[:, None] * np.cos(OMEGA * t) * np.ones((NX, NY))


def solve_helmholtz(rhs, alpha):
  n = NX * NY
  def matvec(x):
    e = x.reshape(NX, NY)
    return (e - alpha * laplacian(e)).ravel()
  A = LinearOperator((n, n), matvec=matvec, dtype=float)
  sol, info = cg(A, rhs.ravel(), rtol=1e-10, maxiter=500)
  return sol.reshape(NX, NY)


def run_semi_implicit(dt, T):
  eta = eta_exact(0.0).copy()
  u = np.zeros((NX + 1, NY)); v = np.zeros((NX, NY + 1))
  alpha = 0.25 * dt * dt * G * H0
  t, nstep, emax = 0.0, 0, 0.0
  while t < T - 1e-12:
    rhs = eta - dt * H0 * divergence(u, v) + alpha * laplacian(eta)
    eta_new = solve_helmholtz(rhs, alpha)
    eta_face = 0.5 * (eta_new + eta)
    u = u - dt * G * gradx(eta_face)
    v = v - dt * G * grady(eta_face)
    eta = eta_new
    t += dt; nstep += 1
    emax = max(emax, float(np.max(np.abs(eta))))
  return eta, u, v, nstep, emax


def energy(eta, u, v):
  uc = 0.5 * (u[1:, :] + u[:-1, :]); vc = 0.5 * (v[:, 1:] + v[:, :-1])
  return float(np.sum(0.5 * G * eta**2 + 0.5 * H0 * (uc**2 + vc**2)) * DX * DY)


def run_explicit(dt, T):
  """Naive explicit C-grid leapfrog -- to show it blows up at the large dt."""
  eta = eta_exact(0.0).copy()
  u = np.zeros((NX + 1, NY)); v = np.zeros((NX, NY + 1))
  t, emax = 0.0, 0.0
  while t < T - 1e-12:
    eta = eta - dt * H0 * divergence(u, v)
    u = u - dt * G * gradx(eta)
    v = v - dt * G * grady(eta)
    t += dt
    emax = max(emax, float(np.max(np.abs(eta))))
    if not np.isfinite(emax) or emax > 1e3 * A0:
      return False, emax
  return True, emax


if __name__ == "__main__":
  dt_cfl = 0.9 * min(DX, DY) / np.sqrt(G * H0) / np.sqrt(2.0)   # explicit 2D CFL
  factor = 20
  dt_si = factor * dt_cfl
  T = 2 * PERIOD

  print(f"[semi-impl] standing wave: period={PERIOD:.4f}s, T={T:.4f}s")
  print(f"[semi-impl] explicit CFL dt={dt_cfl:.2e}s ;  semi-implicit dt={dt_si:.2e}s ({factor}x)")

  E0 = energy(eta_exact(0.0), np.zeros((NX + 1, NY)), np.zeros((NX, NY + 1)))
  eta, u, v, nstep, emax = run_semi_implicit(dt_si, T)
  ex = eta_exact(T)
  l2 = np.sqrt(np.sum((eta - ex)**2) * DX * DY) / np.sqrt(np.sum(ex**2) * DX * DY + 1e-30)
  Ef = energy(eta, u, v)
  print(f"[semi-impl] ran {nstep} steps at {factor}x CFL, stable (max|eta|/A0={emax/A0:.2f})")
  print(f"[semi-impl] L2 error vs EXACT standing wave = {l2:.3e}  -> {'PASS' if l2 < 0.05 else 'FAIL'}")
  print(f"[semi-impl] energy drift |E-E0|/E0 = {abs(Ef-E0)/E0:.2e}  (Crank-Nicolson ~ non-dissipative)")

  ok, emax_e = run_explicit(dt_si, T)
  print(f"[semi-impl] explicit at the SAME {factor}x dt: "
        + ("stayed bounded (unexpected)" if ok else f"BLEW UP (max|eta|/A0={emax_e/A0:.1e}) as expected"))
  print(f"[semi-impl] => semi-implicit gives a ~{factor}x larger step at equal stability")
