#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semi-implicit TWO-LAYER shallow water (barotropic implicit, baroclinic explicit).

This is the multilayer semi-implicit time integration: the fast barotropic
free-surface wave (c0 = sqrt(gH)) is treated implicitly via a Helmholtz solve for
the surface eta_s, while the slow internal (baroclinic) mode (c1 = sqrt(g' H1H2/H),
c1 << c0) stays explicit. The step is then limited only by the SLOW baroclinic wave
-> a ~c0/c1 speedup over a fully explicit scheme, which cannot pass the barotropic
CFL. This is the core that would remove the barotropic bottleneck of the explicit
multilayer HLLC solver (the plunging-plume runs were slow for exactly this reason).

1D staggered grid: eta_s, eta_i (surface & interface deviation) at cell centres,
u1 (bottom), u2 (top) at faces. Linearised Boussinesq two-layer equations:
    d_t u2 = -g d_x eta_s
    d_t u1 = -g d_x eta_s - g' d_x eta_i           (g' = g (rho1-rho2)/rho2)
    d_t eta_i = -H1 d_x u1
    d_t eta_s = -d_x (H1 u1 + H2 u2)
Crank-Nicolson (theta=1/2) on the barotropic (g, gH) terms, explicit on g'.

VALIDATION: an internal standing wave returns to its initial shape after one
baroclinic period T1 = 2*pi/(c1 k); run at dt >> the barotropic CFL (where an
explicit scheme blows up) and check eta_i(T1) ~ eta_i(0).
"""
import numpy as np
from scipy.linalg import solve_banded

G = 9.81
RHO1, RHO2 = 1030., 1000.
GP = G * (RHO1 - RHO2) / RHO2               # reduced gravity
H1, H2 = 0.5, 0.5
H = H1 + H2
LX, NX = 1.0, 100
DX = LX / NX
A0, KX = 1e-3, np.pi / LX
C0 = np.sqrt(G * H)                          # barotropic wave speed (fast)
C1 = np.sqrt(GP * H1 * H2 / H)               # baroclinic wave speed (slow)
OMEGA1 = C1 * KX
T1 = 2 * np.pi / OMEGA1                       # baroclinic period
xc = (np.arange(NX) + 0.5) * DX
THETA = 0.5


def ddx_face(eta):                            # centre -> face, 0 on walls (Neumann)
  g = np.zeros(NX + 1)
  g[1:-1] = (eta[1:] - eta[:-1]) / DX
  return g

def div_cell(u):                              # face -> centre
  return (u[1:] - u[:-1]) / DX

def lap_cell(eta):
  return div_cell(ddx_face(eta))


def helmholtz_band(alpha):
  """Tridiagonal (I - alpha*lap) with Neumann BC, as banded matrix for solve_banded."""
  ab = np.zeros((3, NX))
  c = alpha / DX**2
  for i in range(NX):
    left = 1 if i > 0 else 0
    right = 1 if i < NX - 1 else 0
    ab[1, i] = 1.0 + c * (left + right)       # diagonal
    if i > 0:
      ab[2, i - 1] = -c                        # sub-diagonal (col i-1)
    if i < NX - 1:
      ab[0, i + 1] = -c                        # super-diagonal (col i+1)
  return ab


def run(dt, T, scheme):
  eta_i = A0 * np.cos(KX * xc)
  eta_s = np.zeros(NX)
  u1 = np.zeros(NX + 1); u2 = np.zeros(NX + 1)
  alpha = dt * dt * THETA * THETA * G * H
  ab = helmholtz_band(alpha)
  t, emax = 0.0, 0.0
  while t < T - 1e-12:
    if scheme == "semi":
      Q = H1 * u1 + H2 * u2
      rhs = (eta_s - dt * div_cell(Q)
             + dt * dt * THETA * (1 - THETA) * G * H * lap_cell(eta_s)
             + dt * dt * THETA * H1 * GP * lap_cell(eta_i))
      eta_s_new = solve_banded((1, 1), ab, rhs)
      gs = ddx_face(THETA * eta_s_new + (1 - THETA) * eta_s)
      gi = ddx_face(eta_i)                     # baroclinic explicit
      u1n = u1 - dt * (G * gs + GP * gi)
      u2n = u2 - dt * (G * gs)
      # forward-backward (symplectic): interface uses the FULL new bottom velocity;
      # a theta-average here is weakly unstable for the explicit baroclinic wave.
      eta_i = eta_i - dt * H1 * div_cell(u1n)
      u1, u2, eta_s = u1n, u2n, eta_s_new
    else:                                      # fully explicit (blows up above barotropic CFL)
      eta_i = eta_i - dt * H1 * div_cell(u1)
      eta_s = eta_s - dt * div_cell(H1 * u1 + H2 * u2)
      u1 = u1 - dt * (G * ddx_face(eta_s) + GP * ddx_face(eta_i))
      u2 = u2 - dt * (G * ddx_face(eta_s))
    t += dt
    emax = max(emax, float(np.max(np.abs(eta_i))))
    if not np.isfinite(emax) or emax > 1e3 * A0:
      return None, emax, t
  return eta_i, emax, t


if __name__ == "__main__":
  dt_baro = DX / C0                            # barotropic (fast) CFL
  dt_bcl = DX / C1                             # baroclinic (slow) CFL
  dt = 0.5 * dt_bcl                            # resolve the slow wave; >> barotropic CFL
  ratio = dt / dt_baro

  print(f"[2layer] c0(barotropic)={C0:.3f}  c1(baroclinic)={C1:.3f}  c0/c1={C0/C1:.1f}")
  print(f"[2layer] barotropic CFL dt={dt_baro:.2e}s ; run dt={dt:.2e}s ({ratio:.1f}x barotropic CFL)")
  print(f"[2layer] baroclinic period T1={T1:.3f}s")

  eta_i, emax, t = run(dt, T1, "semi")
  ei0 = A0 * np.cos(KX * xc)
  l2 = np.sqrt(np.sum((eta_i - ei0)**2)) / np.sqrt(np.sum(ei0**2))
  print(f"[2layer] semi-implicit: STABLE at {ratio:.1f}x barotropic CFL (max|eta_i|/A0={emax/A0:.2f})")
  print(f"[2layer] interface returned after 1 baroclinic period: L2||eta_i(T1)-eta_i(0)|| = {l2:.3e}"
        f"  -> {'PASS' if l2 < 0.1 else 'FAIL'}")

  res, emax_e, t_e = run(dt, T1, "explicit")
  print(f"[2layer] explicit at the SAME dt: "
        + ("bounded (unexpected)" if res is not None else f"BLEW UP at t={t_e:.2f}s (max/A0={emax_e/A0:.1e})"))
  print(f"[2layer] => semi-implicit steps at the baroclinic scale, {ratio:.1f}x past the barotropic CFL")
