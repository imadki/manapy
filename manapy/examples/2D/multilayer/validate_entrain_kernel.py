#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic unit test of the entrainment closure kernel (v2).

Validates the Parker/Ellison-Turner exchange kernel directly, independent of the
flow solver (so it is unaffected by any flux instability):
  * entrainment rate w_e == E(Ri)|u|  with E = E0 / sqrt(1 + a Ri^n);
  * mass / salt / momentum exchange is conservative between the two layers;
  * correct limits: E -> E0 as Ri -> 0 (strong shear), E -> 0 as Ri -> inf.
"""
import numpy as np
from manapy.solvers.multilayer import fvm_utils_compute as ml

ml.setup(2)

# two layers, a sweep of bottom-layer speeds -> a sweep of Richardson numbers
rho = np.array([1030., 1000.])
rho0, grav = 1000., 9.81
E0, a_par, n_par = 0.075, 718.0, 2.4
gp = grav * (rho[0] - rho[1]) / rho0          # reduced gravity across the interface
h0 = 0.02                                     # dense layer thickness

u = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
n = len(u)
h_all = np.ascontiguousarray(np.vstack([np.full(n, h0), np.full(n, 0.10)]))
hu_all = np.ascontiguousarray(np.vstack([h0 * u, np.zeros(n)]))
hv_all = np.ascontiguousarray(np.zeros((2, n)))
s_all = np.ascontiguousarray(np.vstack([h0 * np.ones(n), np.zeros(n)]))   # c1=1, c2=0

exch_h = np.zeros((2, n)); exch_hu = np.zeros((2, n))
exch_hv = np.zeros((2, n)); exch_s = np.zeros((2, n))

# large dt + huge cap so the positivity cap never binds (test the raw closure)
ml.entrainment_exchange(h_all, hu_all, hv_all, s_all, rho, grav, rho0, E0, a_par, n_par,
                        1.0, 1e12, exch_h, exch_hu, exch_hv, exch_s)

Ri = gp * h0 / u**2
E_exact = E0 / np.sqrt(1.0 + a_par * Ri**n_par)
we_exact = E_exact * u

err = np.max(np.abs(exch_h[0] - we_exact))
cons_mass = np.max(np.abs(exch_h[0] + exch_h[1]))
cons_salt = np.max(np.abs(exch_s[0] + exch_s[1]))
cons_mom = np.max(np.abs(exch_hu[0] + exch_hu[1]))

print("  u      Ri      E(Ri)     w_e(kernel)  w_e(exact)")
for i in range(n):
  print(f"  {u[i]:.2f}  {Ri[i]:7.3f}  {E_exact[i]:.5f}   {exch_h[0][i]:.3e}   {we_exact[i]:.3e}")

print(f"\n[kernel] max|w_e - E(Ri)|u||      = {err:.2e}   -> {'PASS' if err < 1e-14 else 'FAIL'}")
print(f"[kernel] mass  exchange antisym   = {cons_mass:.2e}   -> {'PASS' if cons_mass < 1e-18 else 'FAIL'}")
print(f"[kernel] salt  exchange antisym   = {cons_salt:.2e}   -> {'PASS' if cons_salt < 1e-18 else 'FAIL'}")
print(f"[kernel] momentum exchange antisym= {cons_mom:.2e}   -> {'PASS' if cons_mom < 1e-18 else 'FAIL'}")
print(f"[kernel] limit Ri->0  E~E0={E0}:  E(Ri={Ri[-1]:.3f})={E_exact[-1]:.4f}  (high-u -> max mixing)")
print(f"[kernel] limit Ri>>1  E->0     :  E(Ri={Ri[0]:.2f})={E_exact[0]:.2e}  (strong stratif -> no mixing)")
