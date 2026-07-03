#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the reverse-osmosis membrane-channel solver (manapy.solvers.ro).

Two layers:
  * unit tests of the membrane closures (pure functions, no mesh);
  * integration tests of the full transient solver on the channel mesh,
    checking the physical trends -- concentration polarisation, salt rejection,
    and fouling-driven flux decline.
"""
import os
import numpy as np
import pytest

from manapy.solvers.ro import membrane as mb

MESH = os.path.join(os.path.dirname(__file__), "..", "meshes", "ro_channel.msh")
FEED = 35.0


# --------------------------------------------------------------------------- #
# Unit tests: membrane closures
# --------------------------------------------------------------------------- #
def test_osmotic_pressure_linear_and_nonnegative():
    c = np.array([-5.0, 0.0, 35.0])
    pi = mb.osmotic_pressure(c, coeff=8.0e4)
    assert pi[0] == 0.0                      # negative conc clipped
    assert pi[1] == 0.0
    assert pi[2] == pytest.approx(8.0e4 * 35.0)


def test_water_flux_decreases_with_fouling():
    """More fouling resistance -> lower permeation velocity, never negative."""
    pi_w = mb.osmotic_pressure(np.array([35.0]), 8.0e4)
    pi_p = np.array([0.0])
    args = dict(dP=6.0e6, pi_w=pi_w, pi_p=pi_p, mu=1e-3,
                R_m=5e14, sigma=1.0)
    Jw_clean = mb.water_flux(R_f=np.array([0.0]), **args)
    Jw_foul = mb.water_flux(R_f=np.array([5e14]), **args)
    assert Jw_clean[0] > Jw_foul[0] > 0.0

    # osmotic pressure exceeding applied pressure -> no back-flux
    Jw_block = mb.water_flux(dP=1.0e5, pi_w=pi_w, pi_p=pi_p, mu=1e-3,
                             R_m=5e14, R_f=np.array([0.0]), sigma=1.0)
    assert Jw_block[0] == 0.0


def test_permeate_concentration_bounds():
    c_w = np.array([35.0, 50.0])
    Jw = np.array([5e-5, 5e-5])
    cp = mb.permeate_conc(c_w, Jw, B=5e-8)
    assert np.all(cp > 0.0)
    assert np.all(cp < c_w)                  # membrane rejects salt


# --------------------------------------------------------------------------- #
# Integration tests: full solver on the channel mesh
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def domain():
    if not os.path.exists(MESH):
        pytest.skip(f"channel mesh not found: {MESH}")
    from manapy.domain import Domain, Partitioning
    return Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)


def _make_solver(domain, **kw):
    from manapy.core.Variable import Variable
    from manapy.solvers.ro import ReverseOsmosisSolver
    c = Variable(domain=domain,
                 BC={"in": "dirichlet", "out": "neumann",
                     "upper": "neumann", "bottom": "neumann"},
                 values_dict={"in": FEED})
    u = Variable(domain=domain)
    v = Variable(domain=domain)
    c.cell[:] = FEED
    params = dict(feed_conc=FEED, U0=0.01, D=1.0e-8,
                  A_w=8.0e-12, B_s=5.0e-8, dP=6.5e6)
    params.update(kw)
    return ReverseOsmosisSolver(c, vel=(u, v), **params)


def test_solver_runs_and_is_physical(domain):
    solver = _make_solver(domain, fouling=True, fouling_coeff=0.4)
    hist = solver.run(nsteps=150, history_every=10)

    # everything finite
    for k, v in hist.items():
        assert np.all(np.isfinite(v)), f"non-finite values in {k}"

    d = solver.diagnostics()
    # salt is rejected: 0 < permeate conc < wall conc
    assert 0.0 < d["cp_mean"] < d["cw_mean"]
    # recovery is a sensible fraction
    assert 0.0 < d["recovery"] < 1.0


def test_concentration_polarisation(domain):
    """Wall concentration must rise above the feed (salt builds up at the wall)."""
    solver = _make_solver(domain, fouling=False)
    solver.run(nsteps=200)
    c_w, Jw, cp = solver._membrane_state()
    # the downstream wall concentration clearly exceeds the feed
    assert c_w.max() > FEED * 1.02
    assert c_w.mean() >= FEED


def test_fouling_declines_flux(domain):
    """With fouling on, permeate flux declines and R_f grows; off, it does not."""
    foul = _make_solver(domain, fouling=True, fouling_coeff=0.4)
    h_foul = foul.run(nsteps=200, history_every=10)

    clean = _make_solver(domain, fouling=False)
    h_clean = clean.run(nsteps=200, history_every=10)

    # fouling case: flux drops substantially and resistance accumulates
    assert h_foul["flux_LMH"][-1] < 0.9 * h_foul["flux_LMH"][0]
    assert h_foul["Rf_over_Rm"][-1] > 0.1
    assert np.all(np.diff(h_foul["Rf_over_Rm"]) >= -1e-12)   # monotone growth

    # clean case: no fouling resistance, flux does not collapse
    assert np.allclose(h_clean["Rf_over_Rm"], 0.0)
    assert h_clean["flux_LMH"][-1] > 0.9 * h_clean["flux_LMH"][0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
