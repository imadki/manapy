#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0-D H2/air autoignition with REAL chemistry via the Cantera backend.

Unlike ignition0d.py (a hand-fed model mechanism that only exercises the engine),
this uses a real H2/O2 mechanism (Cantera's h2o2.yaml: NASA-7 thermodynamics +
literature Arrhenius kinetics) -- a standard 9-species H2/air system.
Cantera is the open-source CHEMKIN: it reads the same
mechanism files and evaluates the same thermo/kinetics.

Run (needs `pip install cantera`):
    python3 ignition_cantera0d.py
"""
import numpy as np

from manapy.solvers.euler.cantera_backend import CanteraChemistry

chem = CanteraChemistry("h2o2.yaml")

# stoichiometric H2/air at constant volume, adiabatic
import cantera as ct
Y = chem.mass_fractions_from(H2=2 * 2.016, O2=32.0, N2=3.76 * 28.0)
T0 = 1100.0
chem.gas.TPY = T0, ct.one_atm, Y
rho = chem.gas.density

t, T, Yhist = chem.react_cell(rho, T0, Y, dt=5.0e-3, history=True)
Teq = chem.equilibrium_T(rho, T0, Y)
tign = t[int(np.argmax(np.gradient(T, t)))]

iH2O = chem.index("H2O")
iH2 = chem.index("H2")
print(f"mechanism: {chem.mechanism}  ({chem.nspec} species, {chem.gas.n_reactions} reactions)")
print(f"initial   T0 = {T0:.0f} K  (stoichiometric H2/air, constant volume)")
print(f"ignition delay (max dT/dt) = {tign:.3e} s")
print(f"final     T = {T[-1]:.0f} K   (UV equilibrium {Teq:.0f} K)")
print(f"          Y_H2 {Yhist[iH2,0]:.4f} -> {Yhist[iH2,-1]:.4f}   "
      f"Y_H2O {Yhist[iH2O,0]:.4f} -> {Yhist[iH2O,-1]:.4f}")

lo, hi = T.min(), T.max()
ramp = " .:-=+*#%@"
spark = "".join(ramp[min(len(ramp) - 1, int((Ti - lo) / (hi - lo + 1e-30) * (len(ramp) - 1)))]
                for Ti in T[::max(1, len(T) // 60)])
print("T(t): " + spark)
