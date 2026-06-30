#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0-D constant-volume autoignition: the reactive chemistry core (Phase 5).

A stoichiometric Fuel + Oxidizer -> Product mixture is held at constant volume and
left to react adiabatically. Arrhenius mass-action kinetics drive an induction
period followed by a thermal runaway (ignition); the temperature climbs to the
adiabatic flame value with total internal energy and mass exactly conserved.

This validates the chemistry kernels (Arrhenius rates + stiff integration) and
the composition-dependent thermodynamics independently of the flow solver.

Run:
    python3 ignition0d.py
"""
import numpy as np

from manapy.solvers.euler.thermo import MixtureThermo, R_UNIVERSAL
from manapy.solvers.euler.chemistry import Chemistry

# --- model mechanism:  F + O -> P  (mass-conserving: W_P = W_F + W_O) ---
W = [2.0, 32.0, 34.0]
e0 = [2.0e7, 0.0, 0.0]            # J/kg formation energy: fuel is energetic
thd = MixtureThermo(W=W, gamma=[1.3, 1.3, 1.3], e0=e0, names=["F", "O", "P"])

chem = Chemistry(thd,
                 nu_reac=[[1, 1, 0]], nu_prod=[[0, 0, 1]],
                 A=[2.0e11], beta=[0.0], Ea=[18000.0 * R_UNIVERSAL])

# stoichiometric reactants by mass
YF = W[0] / (W[0] + W[1])
YO = W[1] / (W[0] + W[1])
Y0 = [YF, YO, 0.0]
rho = 1.0
T0 = 950.0                       # cool enough to show a clear induction period

t_end = 2.0
t, T, Y = chem.react_cell(rho, T0, Y0, t_end, history=True)

# adiabatic flame temperature for the complete reaction
e_const = thd.mixture_cv(Y0) * T0 + thd.chemical_energy(Y0)
Tad = (e_const - thd.chemical_energy([0, 0, 1.0])) / thd.mixture_cv([0, 0, 1.0])

# ignition delay = time of maximum dT/dt
dTdt = np.gradient(T, t)
tign = t[int(np.argmax(dTdt))]

print(f"initial   T0 = {T0:.1f} K,  stoichiometric Y = (F={YF:.4f}, O={YO:.4f})")
print(f"adiabatic flame T = {Tad:.1f} K")
print(f"ignition delay (max dT/dt) = {tign:.3e} s")
print(f"final     T = {T[-1]:.1f} K,  Y = (F={Y[0,-1]:.5f}, O={Y[1,-1]:.5f}, P={Y[2,-1]:.5f})")

# energy & mass conservation across the whole history
e_hist = np.array([thd.mixture_cv(Y[:, i]) * T[i] + thd.chemical_energy(Y[:, i])
                   for i in range(len(t))])
print(f"energy drift over history: {np.max(np.abs(e_hist - e_const)) / e_const:.2e}")
print(f"max |sum(Y)-1| over history: {np.max(np.abs(Y.sum(axis=0) - 1.0)):.2e}")

# coarse text sparkline of T(t)
lo, hi = T.min(), T.max()
ramp = " .:-=+*#%@"
print("T(t): " + "".join(ramp[min(len(ramp) - 1, int((Ti - lo) / (hi - lo + 1e-30) * (len(ramp) - 1)))]
                          for Ti in T[::max(1, len(T) // 60)]))
