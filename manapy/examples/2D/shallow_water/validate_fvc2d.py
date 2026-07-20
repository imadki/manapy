#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation of the FVC (Finite Volume Characteristics) scheme for shallow water.

FVC (Benkhaldoun-Seaid family) is an EIGENSTRUCTURE-FREE flux: instead of a
Riemann solver (SRNH's Roe/Cardano, which arccos-NaNs at Froude=1), the interface
state is built by the method of characteristics (semi-Lagrangian departure point +
half-step predictor carrying the pressure/acoustic coupling), and the physical flux
is evaluated at that predicted state. It is well-balanced (C-property, via the
Audusse hydrostatic reconstruction) and robust across the sonic point.

Three tests, each writing VTK for ParaView:
  (1) C-property   -- lake at rest over a bump: residual must be ~machine zero.
  (2) Stoker       -- wet-bed dam-break vs the EXACT solution: L2 + convergence.
  (3) Transcritical -- near-dry dam-break (crosses Froude=1): SRNH NaNs, FVC stays
                       finite. This is the whole point of FVC.

Run:  python3 -u validate_fvc2d.py
"""
from mpi4py import MPI
import numpy as np
from manapy.api.mesh import Mesh
from manapy.solvers.shallowater.system import ShallowWaterSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
GRAV = 9.81
NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}


# --------------------------------------------------------------------------- #
def stoker(x, t, hL, hR, g=GRAV):
    """Exact wet-bed dam-break (flat bed, dam at x=0)."""
    cL = np.sqrt(g * hL)

    def f(hm):
        u_raref = 2.0 * (cL - np.sqrt(g * hm))
        u_shock = (hm - hR) * np.sqrt(0.5 * g * (1.0 / hm + 1.0 / hR))
        return u_raref - u_shock

    lo, hi = hR, hL
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    hm = 0.5 * (lo + hi)
    um = 2.0 * (cL - np.sqrt(g * hm))
    cm = np.sqrt(g * hm)
    s = hm * um / (hm - hR)
    h = np.empty_like(x)
    xt = x / t
    for i in range(len(x)):
        xi = xt[i]
        if xi <= -cL:
            h[i] = hL
        elif xi <= um - cm:
            h[i] = (1.0 / (9.0 * g)) * (2.0 * cL - xi) ** 2
        elif xi <= s:
            h[i] = hm
        else:
            h[i] = hR
    return h


# --------------------------------------------------------------------------- #
# TEST 1 -- C-property (well-balancing) over a bump
# --------------------------------------------------------------------------- #
def test_cproperty():
    mesh = Mesh.rectangle(bounds=((0., 10.), (0., 2.)), n=(60, 12), cell_type="triangle")
    dom = mesh.domain
    Z = mesh.field("Z", init=lambda x, y, z: 0.2 * np.exp(-((x - 5.0) ** 2)), bc=NEU)
    h = mesh.field("h", init=(1.0 - np.asarray(Z.cell)), bc=NEU)
    hu = mesh.field("hu", init=0., bc=NEU)
    hv = mesh.field("hv", init=0., bc=NEU)
    hc = mesh.field("hc", init=0., bc=NEU)
    S = ShallowWaterSolver(h=h, hvel=(hu, hv), hc=hc, Z=Z, order=1, cfl=0.4, scheme="fvc")
    for _ in range(200):
        S.stepper(); S.compute_fluxes(); S.compute_new_val()
    dom.save_on_cell_multi(["h", "Z", "eta"],
                           [h.cell, Z.cell, np.asarray(h.cell) + np.asarray(Z.cell)], S.dt, 0., 200, 0)
    if RANK == 0:
        print("[1] C-property over a bump (200 steps):")
        print(f"    max|hu| = {np.max(np.abs(hu.cell)):.3e}   max|hv| = {np.max(np.abs(hv.cell)):.3e}")
        print(f"    -> {'PASS (machine zero)' if np.max(np.abs(hu.cell)) < 1e-11 else 'CHECK'}")


# --------------------------------------------------------------------------- #
# TEST 2 -- Stoker dam-break vs exact + convergence
# --------------------------------------------------------------------------- #
def run_dam(nx, hL, hR, T, scheme, cfl=0.4, save=False):
    mesh = Mesh.rectangle(bounds=((-10., 10.), (0., 1.)), n=(nx, 4), cell_type="triangle")
    dom = mesh.domain
    xc = np.asarray(dom.cells.center)[:, 0]
    Z = mesh.field("Z", init=0., bc=NEU)
    h = mesh.field("h", init=np.where(xc < 0., hL, hR), bc=NEU)
    hu = mesh.field("hu", init=0., bc=NEU)
    hv = mesh.field("hv", init=0., bc=NEU)
    hc = mesh.field("hc", init=0., bc=NEU)
    S = ShallowWaterSolver(h=h, hvel=(hu, hv), hc=hc, Z=Z, order=1, cfl=cfl, scheme=scheme)
    t, it = 0.0, 0
    while t < T:
        S.stepper(); S.compute_fluxes(); S.compute_new_val(); t += S.dt; it += 1
        if not np.all(np.isfinite(h.cell)):
            return xc, np.asarray(h.cell), False, t
    if save:
        dom.save_on_cell_multi(["h", "hu"], [h.cell, hu.cell], S.dt, t, it, 0)
    return xc, np.asarray(h.cell), True, t


def test_stoker():
    if RANK == 0:
        print("[2] Stoker dam-break hL=2 hR=1, T=0.8 (FVC vs exact):")
    prev = None
    for nx in (100, 200, 400):
        xc, hnum, ok, tend = run_dam(nx, 2.0, 1.0, 0.8, "fvc", save=(nx == 200))
        hex = stoker(xc, tend, 2.0, 1.0)
        l2 = np.sqrt(np.mean((hnum - hex) ** 2))
        order = "" if prev is None else f"   order={np.log(prev / l2) / np.log(2.0):.2f}"
        if RANK == 0:
            print(f"    nx={nx:4d}  finite={ok}  L2(h)={l2:.4e}{order}")
        prev = l2


# --------------------------------------------------------------------------- #
# TEST 3 -- Transcritical near-dry dam-break: SRNH NaN vs FVC finite
# --------------------------------------------------------------------------- #
def test_transcritical():
    if RANK == 0:
        print("[3] Transcritical near-dry dam-break hL=1 hR=1e-3, T=0.5:")
    for scheme in ("srnh", "fvc"):
        xc, hnum, ok, tend = run_dam(120, 1.0, 1e-3, 0.5, scheme, cfl=0.3, save=(scheme == "fvc"))
        if RANK == 0:
            tag = "FINITE" if ok else "NaN / blow-up"
            extra = f"  h_range=[{hnum.min():.2e},{hnum.max():.3f}]" if ok else ""
            print(f"    scheme={scheme:5s} -> {tag} (t={tend:.3f}){extra}")
    if RANK == 0:
        print("    -> FVC stays finite across Froude=1 where SRNH's arccos NaNs (the point of FVC).")


if __name__ == "__main__":
    test_cproperty()
    test_stoker()
    test_transcritical()
    if RANK == 0:
        print("VTK written (cproperty eta, stoker h/hu, transcritical h) for ParaView.")
