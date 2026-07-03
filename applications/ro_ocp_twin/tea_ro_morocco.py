#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.2 -- TEA localized for OCP / Morocco.

Same SWRO + energy-recovery flowsheet and WaterTAP costing as ``tea_ro.py``, but
with the cost assumptions localized to an OCP Moroccan plant:

  * electricity   : 0.04 $/kWh   -- OCP's dedicated-renewable strategy (Moroccan
                    wind is among the cheapest worldwide); grid (ONEE, ~0.10) and
                    the WaterTAP US default (0.07) are kept as sensitivities;
  * discount rate : WACC 8 %      -- Moroccan infrastructure, strong OCP credit
                    (the WaterTAP default is ~9.3 %; concessional ~6 % as a case);
  * currency      : LCOW reported in $/m3, MAD/m3 and EUR/m3.

Everything else (30-yr life, 90 % utilization, labor/chem/maintenance factor) is
kept at WaterTAP defaults; labor is a small lumped fraction and chemicals are
globally priced, so the dominant local levers are electricity and the WACC.

Note: WaterTAP LCOW is in USD_2018; FX below converts that base (no extra
2018->2026 inflation is applied, per WaterTAP convention).
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

# --- localization parameters ---------------------------------------------- #
ELEC_OCP = 0.04        # $/kWh  OCP dedicated renewable (base case)
ELEC_GRID = 0.10       # $/kWh  Moroccan grid (ONEE industrial, ~1.0 MAD/kWh)
ELEC_US = 0.07         # $/kWh  WaterTAP US default
WACC_MA = 0.08         # Moroccan infrastructure (base case)
WACC_WT = 0.09307      # WaterTAP default
WACC_LOW = 0.06        # concessional / sovereign-backed
USD_MAD = 10.0         # ~2024-2026
EUR_MAD = 10.9         # ~2024-2026  (=> EUR/USD ~ 1.09)

solver = get_solver()


def localize(m, elec, wacc):
    """Apply the localized electricity cost and WACC, then re-solve to optimum."""
    m.fs.costing.electricity_cost.fix(elec)
    m.fs.costing.wacc.fix(wacc)        # capital_recovery_factor recomputes (unfixed)
    solve(m, solver=solver)


def read():
    usd = value(m.fs.costing.LCOW)
    sec = value(m.fs.costing.specific_energy_consumption)
    return dict(
        usd=usd, mad=usd * USD_MAD, eur=usd * USD_MAD / EUR_MAD,
        sec=sec, e_share=sec * value(m.fs.costing.electricity_cost) / usd,
        P_bar=value(m.fs.P1.control_volume.properties_out[0].pressure) / 1e5,
        area=value(m.fs.RO.area),
        crf=value(m.fs.costing.capital_recovery_factor),
    )


# --- build once, optimize at the localized base case ---------------------- #
m = build(erd_type=ERDtype.pressure_exchanger)
set_operating_conditions(m, water_recovery=0.5)
initialize_system(m, solver=solver)
solve(m, solver=solver)
optimize_set_up(m)                  # objective = min LCOW; pump P & area free
localize(m, ELEC_OCP, WACC_MA)
base = read()

print("=" * 74)
print("SWRO TEA localized for OCP / Morocco  (feed 35 g/L, 50% recovery)")
print(f"   base case: elec {ELEC_OCP:.02f} $/kWh (OCP renewable) | WACC {WACC_MA:.0%}"
      f" (CRF {base['crf']:.3f}) | 1$={USD_MAD:.1f} MAD, 1EUR={EUR_MAD:.1f} MAD")
print("=" * 74)
print(f"LCOW  {base['usd']:.3f} $/m3  =  {base['mad']:.2f} MAD/m3  =  {base['eur']:.3f} EUR/m3")
print(f"SEC   {base['sec']:.2f} kWh/m3   (energy = {base['e_share']*100:.0f}% of LCOW)"
      f"   |  P {base['P_bar']:.1f} bar   area {base['area']:.0f} m2")

print("\n1) ELECTRICITY SENSITIVITY  (WACC 8%)")
print(f"{'elec $/kWh':>12}{'source':>14}{'LCOW $/m3':>11}{'MAD/m3':>9}{'EUR/m3':>9}"
      f"{'E-share':>9}")
for elec, src in ((ELEC_OCP, "OCP renew."), (ELEC_US, "WaterTAP"), (ELEC_GRID, "ONEE grid")):
    localize(m, elec, WACC_MA)
    r = read()
    print(f"{elec:>12.2f}{src:>14}{r['usd']:>11.3f}{r['mad']:>9.2f}{r['eur']:>9.3f}"
          f"{r['e_share']*100:>8.0f}%")

print("\n2) DISCOUNT-RATE (WACC) SENSITIVITY  (elec 0.04 $/kWh)")
print(f"{'WACC':>8}{'CRF':>8}{'LCOW $/m3':>11}{'MAD/m3':>9}{'EUR/m3':>9}")
for wacc in (WACC_LOW, WACC_MA, WACC_WT):
    localize(m, ELEC_OCP, wacc)
    r = read()
    print(f"{wacc:>7.1%}{r['crf']:>8.3f}{r['usd']:>11.3f}{r['mad']:>9.2f}{r['eur']:>9.3f}")

print("\n3) FOULING PENALTY  (A_eff = A_clean/(1+R_f/R_m); base case elec+WACC)")
localize(m, ELEC_OCP, WACC_MA)
print(f"{'R_f/R_m':>8}{'LCOW $/m3':>11}{'MAD/m3':>9}{'SEC':>7}{'P bar':>8}{'dLCOW %':>9}")
for Rf in (0.0, 0.25, 0.5, 1.0):
    m.fs.RO.A_comp.fix(A_CLEAN / (1.0 + Rf))
    solve(m, solver=solver)
    r = read()
    print(f"{Rf:>8.2f}{r['usd']:>11.3f}{r['mad']:>9.2f}{r['sec']:>7.2f}{r['P_bar']:>8.1f}"
          f"{(r['usd']/base['usd']-1)*100:>9.1f}")
m.fs.RO.A_comp.fix(A_CLEAN)

print("\n4) RECOVERY TRADE-OFF  (clean membrane; base case elec+WACC)")
print(f"{'recovery':>9}{'LCOW $/m3':>11}{'MAD/m3':>9}{'SEC':>7}{'P bar':>8}{'area m2':>9}")
for r_target in (0.40, 0.50, 0.60):
    m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].fix(r_target)
    solve(m, solver=solver)
    r = read()
    print(f"{r_target*100:>8.0f}%{r['usd']:>11.3f}{r['mad']:>9.2f}{r['sec']:>7.2f}"
          f"{r['P_bar']:>8.1f}{r['area']:>9.0f}")
print("=" * 74)
