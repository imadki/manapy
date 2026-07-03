#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated digital-twin forecast: manapy fouling physics -> WaterTAP economics.

Step A (manapy)  : run the high-fidelity RO solver with fouling and record the
                   fouling-resistance progression R_f/R_m.
Step B (WaterTAP): for each fouling level, map it to an effective membrane
                   permeability  A_eff = A_clean / (1 + R_f/R_m), re-optimise the
                   SWRO stage (min LCOW at fixed recovery) and read the cost and
                   specific energy consumption.

The result is a trajectory  fouling -> {pressure, SEC, LCOW}  -- exactly what a
digital twin needs to forecast when cleaning (CIP) becomes economically
justified.

Run (from anywhere):
    python3 digital_twin_forecast.py
Requires: manapy (this repo) + watertap/idaes + ipopt (idaes get-extensions).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)            # make the local watertap_solvers shim importable

# WaterTAP is used from its source checkout (not pip-installed); add it to the path.
WATERTAP_REPO = os.environ.get(
    "WATERTAP_REPO", os.path.expanduser("~/Documents/GITHUB/watertap"))
if os.path.isdir(WATERTAP_REPO):
    sys.path.insert(0, WATERTAP_REPO)

MESH = os.path.join(HERE, "..", "..", "meshes", "ro_channel.msh")
A_CLEAN = 4.2e-12                   # clean membrane water permeability [m/(s.Pa)]
FEED = 35.0                         # seawater salinity [kg/m3]
CIP_LCOW_THRESHOLD = 10.0          # recommend cleaning when LCOW rises > 10%


# --------------------------------------------------------------------------- #
# Step A — manapy fouling trajectory
# --------------------------------------------------------------------------- #
def manapy_fouling_trajectory(n_samples=5, nsteps=400):
    from manapy.domain import Domain, Partitioning
    from manapy.core.Variable import Variable
    from manapy.solvers.ro import ReverseOsmosisSolver

    dom = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)
    c = Variable(domain=dom, BC={"in": "dirichlet", "out": "neumann",
                                 "upper": "neumann", "bottom": "neumann"},
                 values_dict={"in": FEED})
    u = Variable(domain=dom)
    v = Variable(domain=dom)
    c.cell[:] = FEED
    s = ReverseOsmosisSolver(c, vel=(u, v), feed_conc=FEED, U0=0.01, D=1.0e-8,
                             A_w=A_CLEAN, B_s=5.0e-8, dP=6.5e6,
                             fouling=True, fouling_coeff=0.4)
    hist = s.run(nsteps=nsteps, history_every=max(1, nsteps // (4 * n_samples)))

    # sample n_samples points evenly across the fouling progression
    idx = np.linspace(0, len(hist["Rf_over_Rm"]) - 1, n_samples).astype(int)
    return [(float(hist["time"][i]), float(hist["Rf_over_Rm"][i])) for i in idx]


# --------------------------------------------------------------------------- #
# Step B — WaterTAP techno-economics for a given fouling level
# --------------------------------------------------------------------------- #
def build_tea():
    import idaes  # noqa: F401
    from watertap.core.solvers import get_solver
    from watertap.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (
        build, set_operating_conditions, initialize_system, solve,
        optimize_set_up, ERDtype,
    )
    solver = get_solver()
    m = build(erd_type=ERDtype.pressure_exchanger)
    set_operating_conditions(m, water_recovery=0.5)
    initialize_system(m, solver=solver)
    solve(m, solver=solver)
    optimize_set_up(m)                 # min LCOW; pump pressure & area free
    solve(m, solver=solver)
    return m, solver, solve


def tea_at_fouling(m, solver, solve_fn, Rf_over_Rm):
    from pyomo.environ import value
    m.fs.RO.A_comp.fix(A_CLEAN / (1.0 + Rf_over_Rm))
    solve_fn(m, solver=solver)
    return dict(
        LCOW=value(m.fs.costing.LCOW),
        SEC=value(m.fs.costing.specific_energy_consumption),
        P_bar=value(m.fs.P1.control_volume.properties_out[0].pressure) / 1e5,
    )


# --------------------------------------------------------------------------- #
def main():
    print("Step A: manapy fouling simulation ...")
    traj = manapy_fouling_trajectory()

    print("Step B: WaterTAP techno-economic evaluation ...")
    m, solver, solve_fn = build_tea()
    base = tea_at_fouling(m, solver, solve_fn, 0.0)

    print("\n" + "=" * 78)
    print("DIGITAL-TWIN FORECAST  --  fouling progression -> plant economics")
    print("=" * 78)
    print(f"Clean baseline: LCOW {base['LCOW']:.3f} $/m3 | "
          f"SEC {base['SEC']:.2f} kWh/m3 | P {base['P_bar']:.1f} bar")
    print("-" * 78)
    print(f"{'progression':>12s}{'R_f/R_m':>9s}{'A/A_clean':>11s}"
          f"{'P bar':>8s}{'SEC':>8s}{'LCOW':>8s}{'dLCOW %':>9s}  action")
    cip_flagged = False
    for (t, rf) in traj:
        r = tea_at_fouling(m, solver, solve_fn, rf)
        dl = (r["LCOW"] / base["LCOW"] - 1) * 100
        action = ""
        if dl > CIP_LCOW_THRESHOLD and not cip_flagged:
            action = "<-- recommend CIP (cleaning)"
            cip_flagged = True
        print(f"{t:>12.3g}{rf:>9.3f}{1/(1+rf):>11.3f}"
              f"{r['P_bar']:>8.1f}{r['SEC']:>8.2f}{r['LCOW']:>8.3f}{dl:>9.1f}  {action}")
    print("=" * 78)
    print("The twin converts resolved fouling physics into an operating-cost")
    print("forecast, and flags when cleaning is economically justified.")


if __name__ == "__main__":
    main()
