#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membrane transport closures for the reverse-osmosis solver.

These are the physico-chemical relations that turn a local wall concentration
``c_w`` (and the current fouling resistance ``R_f``) into a permeation velocity
``Jw`` and a permeate concentration ``c_p``.  They are written as plain,
vectorised NumPy functions so they can be evaluated over all membrane faces at
once and are trivially unit-testable in isolation from the finite-volume solver.

Model
-----
* Osmotic pressure (van 't Hoff, linearised):      pi(c)   = phi * c
* Water flux (solution-diffusion, resistance       Jw      = (dP - sigma*dPi)
  in series, fouling layer adds R_f):                        / (mu * (R_m + R_f))
* Salt flux / permeate concentration:              J_s     = B * (c_w - c_p)
                                                            = Jw * c_p
  =>                                                c_p     = B*c_w / (Jw + B)
"""
import numpy as np


def osmotic_pressure(c, coeff):
    """van 't Hoff osmotic pressure [Pa] for concentration ``c`` [kg/m3]."""
    return coeff * np.maximum(c, 0.0)


def water_flux(dP, pi_w, pi_p, mu, R_m, R_f, sigma=1.0):
    """Permeation (water) velocity [m/s] through a fouled membrane.

    Resistance-in-series solution-diffusion model.  The flux is clipped at zero
    so the membrane never lets water flow back into the channel when the local
    osmotic pressure exceeds the applied pressure.
    """
    driving = dP - sigma * (pi_w - pi_p)
    Jw = driving / (mu * (R_m + R_f))
    return np.maximum(Jw, 0.0)


def permeate_conc(c_w, Jw, B):
    """Permeate-side concentration [kg/m3] from the wall solute balance."""
    return B * c_w / (Jw + B + 1e-300)


def membrane_resistance(A_w, mu):
    """Intrinsic (clean) membrane resistance R_m [1/m] from permeability A_w."""
    return 1.0 / (mu * A_w)
