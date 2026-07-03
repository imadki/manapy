#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Techno-economic analysis (TEA) for an SWRO stage with energy recovery.

Built on WaterTAP's costing package (LCOW + specific energy consumption):
  1. base techno-economic optimum (min LCOW at 50% recovery);
  2. FOULING PENALTY  (A_eff = A_clean / (1 + R_f/R_m));
  3. recovery trade-off (energy vs water production).
"""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)            # local watertap_solvers shim
WATERTAP_REPO = os.environ.get(
    "WATERTAP_REPO", os.path.expanduser("~/Documents/GITHUB/watertap"))
if os.path.isdir(WATERTAP_REPO):
    sys.path.insert(0, WATERTAP_REPO)

import idaes  # noqa: E402  registers ipopt
from pyomo.environ import value  # noqa: E402
from watertap.core.solvers import get_solver  # noqa: E402
from watertap.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (  # noqa: E402
    build, set_operating_conditions, initialize_system, solve,
    optimize_set_up, ERDtype,
)

A_CLEAN = 4.2e-12
solver = get_solver()

m = build(erd_type=ERDtype.pressure_exchanger)
set_operating_conditions(m, water_recovery=0.5)
initialize_system(m, solver=solver)
solve(m, solver=solver)
optimize_set_up(m)                  # objective = min LCOW; pump P & area free
solve(m, solver=solver)


def read():
    return dict(
        LCOW=value(m.fs.costing.LCOW),
        SEC=value(m.fs.costing.specific_energy_consumption),
        P_bar=value(m.fs.P1.control_volume.properties_out[0].pressure) / 1e5,
        area=value(m.fs.RO.area),
    )


print("=" * 70)
print("SWRO techno-economic analysis (WaterTAP costing) -- feed 35 g/L, 50% rec.")
print("=" * 70)
base = read()
print(f"Base optimum : LCOW {base['LCOW']:.3f} $/m3 | SEC {base['SEC']:.2f} kWh/m3"
      f" | P {base['P_bar']:.1f} bar | area {base['area']:.0f} m2")

print("\n1) FOULING PENALTY  (A_eff = A_clean / (1 + R_f/R_m))")
print(f"{'R_f/R_m':>8s}{'A/A_clean':>11s}{'LCOW $/m3':>12s}{'SEC kWh/m3':>12s}"
      f"{'P bar':>8s}{'dLCOW %':>9s}")
for Rf in (0.0, 0.25, 0.5, 1.0):
    m.fs.RO.A_comp.fix(A_CLEAN / (1.0 + Rf))
    solve(m, solver=solver)
    r = read()
    print(f"{Rf:>8.2f}{1/(1+Rf):>11.3f}{r['LCOW']:>12.3f}{r['SEC']:>12.2f}"
          f"{r['P_bar']:>8.1f}{(r['LCOW']/base['LCOW']-1)*100:>9.1f}")
m.fs.RO.A_comp.fix(A_CLEAN)

print("\n2) RECOVERY TRADE-OFF  (clean membrane)")
print(f"{'recovery':>9s}{'LCOW $/m3':>12s}{'SEC kWh/m3':>12s}{'P bar':>8s}{'area m2':>9s}")
for r_target in (0.40, 0.50, 0.60):
    m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].fix(r_target)
    solve(m, solver=solver)
    d = read()
    print(f"{r_target*100:>8.0f}%{d['LCOW']:>12.3f}{d['SEC']:>12.2f}"
          f"{d['P_bar']:>8.1f}{d['area']:>9.0f}")
print("=" * 70)
