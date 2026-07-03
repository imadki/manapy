#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.4 -- life-cycle footprint of the SWRO plant (energy, CO2, chemicals, brine).

A transparent *process-LCA* footprint model.  A full ecoinvent/Brightway2 study
needs the (licensed) ecoinvent database, which is not available here; instead we
use published life-cycle emission factors (cited inline) driven by the plant's
own operating point (SEC and recovery from the WaterTAP twin, `tea_ro_morocco.py`).

The point of Phase 2.4 for OCP: the electricity choice of Phase 2.2 is also the
dominant *carbon* lever -- powering the plant with OCP's dedicated renewables
collapses the footprint by ~15x versus the (coal-heavy) Moroccan grid.

Scope: gate-to-gate operating footprint per m3 of permeate
  * energy    : SEC * grid carbon intensity   (scenario-dependent)
  * chemicals : pretreatment + CIP dosing * embodied GWP
  * materials : membranes / cartridge filters (amortized, small)
  * brine     : volume, salinity and salt-discharge load (mass balance)
Not included: plant construction civil works, transport, end-of-life (minor for
an operating-phase comparison; flagged for a future full Brightway2 study).
"""
import numpy as np

# --- plant operating point (from the localized WaterTAP twin) -------------- #
FEED_GL = 35.0            # feed salinity [g/L]
RECOVERY = 0.50           # water recovery [-]
SEC_CLEAN = 2.43          # kWh/m3 permeate, tea_ro_morocco.py base (OCP, WACC 8%)
# SEC rises with fouling (tea_ro_morocco.py fouling block, A_eff=A/(1+R_f/R_m)):
SEC_FOULING = {0.0: 2.43, 0.25: 2.56, 0.5: 2.66, 1.0: 2.82}

# --- electricity carbon intensity [kgCO2eq/kWh, lifecycle] ----------------- #
# Morocco grid: coal-heavy, ~0.6-0.7 (IEA/IFI Morocco grid factor ~2019); OCP
# renewable: onshore wind ~0.011 (IPCC AR5 median), PV ~0.045 -> wind-led ~0.02;
# WaterTAP generic default = 0.475 (watertap_costing_package electrical_carbon_intensity).
CI_ELEC = {"OCP renewable": 0.02, "Morocco grid": 0.65, "WaterTAP default": 0.475}

# --- chemical inventory: dose [g per m3 FEED] and embodied GWP [kgCO2eq/kg] - #
# Representative SWRO dosing + literature cradle-to-gate GWP (e.g. Vince 2008,
# Raluy 2005, Zhou 2011 SWRO LCAs; ecoinvent-order-of-magnitude factors).
CHEMICALS = {
    "antiscalant (phosphonate)": (2.0, 2.0),
    "coagulant (FeCl3)":         (5.0, 1.6),
    "dechlor (Na-bisulfite)":    (4.0, 0.65),
    "CIP (citric/NaOH, amort.)": (1.0, 1.5),
}
GWP_MATERIALS = 0.03      # kgCO2eq/m3, membranes + cartridge filters amortized (~literature)


def chemical_footprint():
    """kgCO2eq per m3 PERMEATE from chemicals (feed->permeate via 1/recovery)."""
    g_per_m3_perm = (1.0 / RECOVERY)        # m3 feed per m3 permeate
    co2 = sum(dose * g_per_m3_perm * 1e-3 * gwp for dose, gwp in CHEMICALS.values())
    mass = sum(dose * g_per_m3_perm for dose, _ in CHEMICALS.values())
    return co2, mass                         # kgCO2/m3 , g chemical/m3


def brine_metrics():
    """Brine volume ratio, salinity and salt-discharge load per m3 permeate."""
    vol_ratio = (1.0 - RECOVERY) / RECOVERY          # m3 brine per m3 permeate
    salinity = FEED_GL / (1.0 - RECOVERY)            # g/L (full rejection limit)
    feed_salt = (1.0 / RECOVERY) * FEED_GL           # g salt in / m3 permeate
    perm_salt = 0.3 * 1.0                             # ~0.3 g/L permeate * volume
    brine_salt = feed_salt - perm_salt               # g salt to sea / m3 permeate
    return vol_ratio, salinity, brine_salt


def main():
    chem_co2, chem_mass = chemical_footprint()
    vol_ratio, brine_sal, brine_salt = brine_metrics()

    print("=" * 74)
    print("SWRO life-cycle footprint (process-LCA)  -- per m3 permeate")
    print(f"   feed {FEED_GL:.0f} g/L | recovery {RECOVERY:.0%} | SEC {SEC_CLEAN:.2f} kWh/m3"
          " (clean, OCP base)")
    print("=" * 74)

    print("CARBON FOOTPRINT by electricity scenario [kgCO2eq / m3]")
    print(f"{'scenario':>20}{'CI kg/kWh':>11}{'energy':>9}{'chem':>8}{'matl':>8}"
          f"{'TOTAL':>9}")
    base_total = None
    for name, ci in CI_ELEC.items():
        e = SEC_CLEAN * ci
        tot = e + chem_co2 + GWP_MATERIALS
        if base_total is None:
            base_total = tot
        print(f"{name:>20}{ci:>11.3f}{e:>9.3f}{chem_co2:>8.3f}{GWP_MATERIALS:>8.3f}"
              f"{tot:>9.3f}")
    grid = SEC_CLEAN * CI_ELEC["Morocco grid"] + chem_co2 + GWP_MATERIALS
    ocp = SEC_CLEAN * CI_ELEC["OCP renewable"] + chem_co2 + GWP_MATERIALS
    print(f"  => OCP renewables cut the carbon footprint {grid/ocp:.0f}x vs the grid "
          f"({grid:.2f} -> {ocp:.2f} kgCO2/m3).")

    print(f"\nCHEMICALS: {chem_mass:.1f} g/m3 permeate, {chem_co2:.3f} kgCO2eq/m3")
    for k, (dose, gwp) in CHEMICALS.items():
        print(f"   {k:<28} {dose/RECOVERY:>6.1f} g/m3   {dose/RECOVERY*1e-3*gwp:>7.4f} kgCO2/m3")

    print(f"\nBRINE: {vol_ratio:.2f} m3/m3 permeate | salinity {brine_sal:.0f} g/L "
          f"(~{brine_sal/FEED_GL:.1f}x feed) | salt load {brine_salt/1000:.3f} kg/m3 permeate")
    print("   management: diffuser dilution to <~5% over ambient salinity; OCP option:")
    print("   route brine to salt/mineral recovery (phosphate-industry synergy).")

    print("\nFOULING -> CARBON  (SEC rises as A_eff = A/(1+R_f/R_m); OCP-renewable power)")
    print(f"{'R_f/R_m':>8}{'SEC':>7}{'energy':>9}{'TOTAL kgCO2/m3':>16}{'dCO2 %':>9}")
    base = None
    for rf, sec in SEC_FOULING.items():
        tot = sec * CI_ELEC["OCP renewable"] + chem_co2 + GWP_MATERIALS
        if base is None:
            base = tot
        print(f"{rf:>8.2f}{sec:>7.2f}{sec*CI_ELEC['OCP renewable']:>9.3f}{tot:>16.3f}"
              f"{(tot/base-1)*100:>9.1f}")
    print("=" * 74)
    print("Note: operating-phase, process-LCA with literature factors; a full")
    print("Brightway2/ecoinvent study (construction, transport, EoL) is future work.")


if __name__ == "__main__":
    main()
