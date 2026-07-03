#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the manapy RO membrane flux law against WaterTAP's ReverseOsmosis0D.

Solves a WaterTAP RO_0D operating point (concentration polarization disabled),
then configures the manapy solver to the SAME membrane and compares the water
flux predicted by the solution-diffusion law:
  * with an INDEPENDENT van 't Hoff osmotic model  (true cross-check), and
  * with the osmotic coefficient MATCHED to WaterTAP (consistency check).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)            # local watertap_solvers shim
WATERTAP_REPO = os.environ.get(
    "WATERTAP_REPO", os.path.expanduser("~/Documents/GITHUB/watertap"))
if os.path.isdir(WATERTAP_REPO):
    sys.path.insert(0, WATERTAP_REPO)
MESH = os.path.join(HERE, "..", "..", "meshes", "ro_channel.msh")

# ---- operating point (matches watertap unit-model test) ------------------- #
FEED_FLOW_MASS, FEED_FRAC = 1.0, 0.035
FEED_P, FEED_T, P_ATM = 50e5, 273.15 + 25, 101325.0
DP, AREA, A, B = 3e5, 50.0, 4.2e-12, 3.5e-8


def watertap_point():
    import idaes  # noqa: F401
    from pyomo.environ import ConcreteModel, value
    from idaes.core import FlowsheetBlock
    from idaes.core.util.scaling import calculate_scaling_factors
    from watertap.core.solvers import get_solver
    from watertap.unit_models.reverse_osmosis_0D import (
        ReverseOsmosis0D, ConcentrationPolarizationType, MassTransferCoefficient)
    import watertap.property_models.NaCl_prop_pack as props

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.NaClParameterBlock()
    u = m.fs.unit = ReverseOsmosis0D(
        property_package=m.fs.properties, has_pressure_change=True,
        concentration_polarization_type=ConcentrationPolarizationType.none,
        mass_transfer_coefficient=MassTransferCoefficient.none)
    u.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(FEED_FLOW_MASS * FEED_FRAC)
    u.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(FEED_FLOW_MASS * (1 - FEED_FRAC))
    u.inlet.pressure[0].fix(FEED_P)
    u.inlet.temperature[0].fix(FEED_T)
    u.deltaP.fix(-DP)
    u.area.fix(AREA)
    u.A_comp.fix(A)
    u.B_comp.fix(B)
    u.permeate.pressure[0].fix(P_ATM)
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "NaCl"))
    calculate_scaling_factors(m)
    u.initialize()
    get_solver().solve(m)
    pin = u.feed_side.properties_in[0]
    return dict(
        feed=value(pin.conc_mass_phase_comp["Liq", "NaCl"]),
        cp=value(u.mixed_permeate[0].conc_mass_phase_comp["Liq", "NaCl"]),
        Jw_in=value(u.flux_mass_phase_comp[0, 0.0, "Liq", "H2O"]) / 1000.0,
        dP_net=FEED_P - P_ATM)


def manapy_inlet_flux(feed, dP, osm_coeff, U0=1.0, nsteps=120):
    from manapy.domain import Domain, Partitioning
    from manapy.core.Variable import Variable
    from manapy.solvers.ro import ReverseOsmosisSolver
    dom = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)
    c = Variable(domain=dom, BC={"in": "dirichlet", "out": "neumann",
                                 "upper": "neumann", "bottom": "neumann"},
                 values_dict={"in": feed})
    u, v = Variable(domain=dom), Variable(domain=dom)
    c.cell[:] = feed
    s = ReverseOsmosisSolver(c, vel=(u, v), feed_conc=feed, U0=U0, D=1e-8,
                             A_w=A, B_s=B, dP=dP, osmotic_coeff=osm_coeff, fouling=False)
    s.run(nsteps=nsteps)
    cw, Jw, _ = s._membrane_state()
    inlet = np.argsort(s.xf[s.mface])[:5]
    return float(Jw[inlet].mean())


def main():
    wt = watertap_point()
    dPi = wt["dP_net"] - wt["Jw_in"] / A             # WaterTAP effective osm. diff
    coeff_match = dPi / (wt["feed"] - wt["cp"])
    coeff_vh = 2 * 8.314 * 298.15 / 0.05844          # van 't Hoff NaCl

    Jw_vh = manapy_inlet_flux(wt["feed"], wt["dP_net"], coeff_vh)
    Jw_m = manapy_inlet_flux(wt["feed"], wt["dP_net"], coeff_match)
    LMH = lambda j: j * 1000 * 3600

    print("=" * 64)
    print("RO flux-law validation : WaterTAP RO_0D vs manapy")
    print("=" * 64)
    print(f"feed {wt['feed']:.2f} kg/m3 | dP_net {wt['dP_net']/1e5:.1f} bar | "
          f"osmotic {dPi/1e5:.1f} bar")
    print(f"{'inlet water flux [LMH]':28s}{'WaterTAP':>11s}"
          f"{'manapy(vH)':>12s}{'manapy(match)':>14s}")
    print(f"{'':28s}{LMH(wt['Jw_in']):>11.2f}{LMH(Jw_vh):>12.2f}{LMH(Jw_m):>14.2f}")
    print(f"{'rel. diff vs WaterTAP [%]':28s}{'-':>11s}"
          f"{(Jw_vh/wt['Jw_in']-1)*100:>12.2f}{(Jw_m/wt['Jw_in']-1)*100:>14.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
