#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sod shock tube validation: manapy compressible Euler vs the EXACT Riemann solution.

The exact Riemann (Toro) solution is the canonical reference for the compressible
Euler core that underpins every phase. Prints volume-weighted L2
errors for rho, u, p for both numerical fluxes (Roe should beat Rusanov).

Run: python3 sod_exact2d.py
"""
import numpy as np
from mpi4py import MPI
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.euler.system import EulerSolver

gamma = 1.4
# standard Sod states
rhoL, uL, pL = 1.0, 0.0, 1.0
rhoR, uR, pR = 0.125, 0.0, 0.1
x0 = 0.5
tend = 0.2


# ---------- exact Riemann solver (Toro) ----------
def exact_sod(x, t):
  g = gamma
  cL = np.sqrt(g * pL / rhoL); cR = np.sqrt(g * pR / rhoR)

  def f(p, rk, pk, ck):
    if p > pk:                       # shock
      A = 2.0 / ((g + 1) * rk); B = (g - 1) / (g + 1) * pk
      return (p - pk) * np.sqrt(A / (p + B))
    else:                            # rarefaction
      return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

  def fp(p):
    return f(p, rhoL, pL, cL) + f(p, rhoR, pR, cR) + (uR - uL)

  # Newton for p_star
  p = 0.5 * (pL + pR)
  for _ in range(100):
    dp = 1e-6 * p
    der = (fp(p + dp) - fp(p - dp)) / (2 * dp)
    pnew = p - fp(p) / der
    if abs(pnew - p) < 1e-12 * p:
      p = pnew; break
    p = max(pnew, 1e-9)
  pstar = p
  ustar = 0.5 * (uL + uR) + 0.5 * (f(pstar, rhoR, pR, cR) - f(pstar, rhoL, pL, cL))

  rho = np.zeros_like(x); u = np.zeros_like(x); pr = np.zeros_like(x)
  for i, xi in enumerate(x):
    s = (xi - x0) / t
    if s < ustar:                    # left of contact
      if pstar > pL:                 # left shock
        rhostarL = rhoL * ((pstar / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pL + 1))
        SL = uL - cL * np.sqrt((g + 1) / (2 * g) * pstar / pL + (g - 1) / (2 * g))
        if s < SL:
          rho[i], u[i], pr[i] = rhoL, uL, pL
        else:
          rho[i], u[i], pr[i] = rhostarL, ustar, pstar
      else:                          # left rarefaction
        rhostarL = rhoL * (pstar / pL) ** (1 / g)
        cstarL = cL * (pstar / pL) ** ((g - 1) / (2 * g))
        SHL = uL - cL; STL = ustar - cstarL
        if s < SHL:
          rho[i], u[i], pr[i] = rhoL, uL, pL
        elif s > STL:
          rho[i], u[i], pr[i] = rhostarL, ustar, pstar
        else:
          uu = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
          cc = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
          rho[i] = rhoL * (cc / cL) ** (2 / (g - 1))
          u[i] = uu; pr[i] = pL * (cc / cL) ** (2 * g / (g - 1))
    else:                            # right of contact
      if pstar > pR:                 # right shock
        rhostarR = rhoR * ((pstar / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pR + 1))
        SR = uR + cR * np.sqrt((g + 1) / (2 * g) * pstar / pR + (g - 1) / (2 * g))
        if s > SR:
          rho[i], u[i], pr[i] = rhoR, uR, pR
        else:
          rho[i], u[i], pr[i] = rhostarR, ustar, pstar
      else:                          # right rarefaction
        rhostarR = rhoR * (pstar / pR) ** (1 / g)
        cstarR = cR * (pstar / pR) ** ((g - 1) / (2 * g))
        SHR = uR + cR; STR = ustar + cstarR
        if s > SHR:
          rho[i], u[i], pr[i] = rhoR, uR, pR
        elif s < STR:
          rho[i], u[i], pr[i] = rhostarR, ustar, pstar
        else:
          uu = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
          cc = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
          rho[i] = rhoR * (cc / cR) ** (2 / (g - 1))
          u[i] = uu; pr[i] = pR * (cc / cR) ** (2 * g / (g - 1))
  return rho, u, pr


# ---------- manapy run ----------
dom = Domain.create_domain('meshes/geo/carre.msh', 2, Partitioning.Par_Nodal, recreate=True)
cells = dom.cells
xc = cells.center[:, 0]
rho = Variable(domain=dom); P = Variable(domain=dom)
rhou = Variable(domain=dom); rhov = Variable(domain=dom); rhoE = Variable(domain=dom)
left = xc < x0
rho.cell[:] = np.where(left, rhoL, rhoR)
P.cell[:] = np.where(left, pL, pR)
rhou.cell[:] = 0.0; rhov.cell[:] = 0.0
rhoE.cell[:] = P.cell / (gamma - 1.0)

for scheme in ['rusanov', 'Roe']:
  rho.cell[:] = np.where(left, rhoL, rhoR)
  P.cell[:] = np.where(left, pL, pR)
  rhou.cell[:] = 0.0; rhov.cell[:] = 0.0
  rhoE.cell[:] = P.cell / (gamma - 1.0)
  S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=gamma, cfl=0.4, scheme=scheme, bc='TubeSchok')
  t = 0.0
  while t < tend:
    dt = S.stepper()
    if t + dt > tend:
      dt = tend - t; S.dt = dt
    t += dt
    S.compute_fluxes(t); S.compute_new_val()
  # compare to exact at cell centres
  re, ue, pe = exact_sod(xc, tend)
  un = rhou.cell / rho.cell
  vol = cells.volume
  def l2(num, ex):
    return np.sqrt(np.sum(vol * (num - ex) ** 2) / np.sum(vol))
  print(f"[{scheme:7s}] L2 errors vs EXACT Sod:  rho={l2(rho.cell,re):.4e}  u={l2(un,ue):.4e}  p={l2(P.cell,pe):.4e}")
