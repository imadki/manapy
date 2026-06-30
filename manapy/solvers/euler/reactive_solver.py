#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reactive compressible flow solver via Strang operator splitting.

Couples three already-validated pieces:
  * EulerSolver        -- compressible hydrodynamics (advection of mass, momentum, energy)
  * SpeciesTransport   -- convective transport of the N species partial densities
  * CanteraChemistry   -- real multispecies thermodynamics (EOS) + finite-rate source

Strang splitting over a hydro step dt:        R(dt/2)  H(dt)  R(dt/2)
  - R: per-cell constant-volume adiabatic reaction (Cantera reactor). It conserves
       rho, momentum and total energy rhoE, so the hydro conservatives are
       untouched; only the composition Y (and hence T, P) change.
  - H: one explicit hydro step + species advection.
After each sub-step the pressure field is refreshed from the *real* equation of
state P = P(rho, e_internal, Y) (Cantera), overwriting the constant-gamma value
the hydro kernel writes. The hydro fluxes still use a representative constant
gamma for the acoustic wave speed (a first-order coupling; variable-gamma fluxes
are the next refinement).

Species order follows the Cantera mechanism (chem.names).
"""
import numpy as np

from manapy.solvers.euler.species import SpeciesTransport


class ReactiveSolver:

  def __init__(self, solver, chemistry, Y0, diffusion=False):
    """
    solver    : EulerSolver (hydro; pass a representative constant gamma)
    chemistry : CanteraChemistry (real thermo + kinetics)
    Y0        : initial mass fractions, list/array length chem.nspec (scalar or
                per-cell), ordered as chemistry.names.
    diffusion : if True, add Fickian species diffusion (mixture-averaged D_k from
                Cantera) each step -- required for propagating premixed flames,
                where heat and radicals diffuse into the unburnt gas.
    """
    self.solver = solver
    self.chem = chemistry
    self.dim = solver.dim
    self.diffusion = bool(diffusion)
    if len(Y0) != chemistry.nspec:
      raise ValueError("Y0 length must equal the number of mechanism species")
    # species transport carries the partial densities, in mechanism order
    self.species = SpeciesTransport(solver, Y0, names=chemistry.names, renormalize=True)
    self._Dk = None
    self._refresh_pressure()

  # --- helpers on the current cell field ---
  def _kinetic_energy(self):
    s = self.solver
    rho = s.rho.cell
    ke = s.rhou.cell ** 2 + s.rhov.cell ** 2
    if self.dim == 3:
      ke = ke + s.rhow.cell ** 2
    return 0.5 * ke / rho

  def _mass_fractions(self):
    rho = self.solver.rho.cell
    return np.column_stack([qk.cell / rho for qk in self.species.q])

  def _refresh_pressure(self):
    """Overwrite P (and, if enabled, the per-cell gamma) from the real EOS."""
    s = self.solver
    rho = s.rho.cell
    e_int = (s.rhoE.cell - self._kinetic_energy()) / rho
    Y = self._mass_fractions()
    _, P, gamma = self.chem.eos_array(rho, e_int, Y)
    s.P.cell[:] = P
    if getattr(s, "variable_gamma", False):
      s.set_gamma(gamma)

  def _react(self, dt_r):
    """Constant-volume reaction sub-step over dt_r; updates Y, T, P (rhoE fixed)."""
    s = self.solver
    rho = s.rho.cell
    e_int = (s.rhoE.cell - self._kinetic_energy()) / rho
    Y = self._mass_fractions()
    Ynew, _ = self.chem.react_array(rho, e_int, Y, dt_r)
    for k, qk in enumerate(self.species.q):
      qk.cell[:] = rho * Ynew[:, k]
    # rhoE unchanged by constant-volume reaction; refresh pressure from new Y
    _, P, gamma = self.chem.eos_array(rho, e_int, Ynew)
    s.P.cell[:] = P
    if getattr(s, "variable_gamma", False):
      s.set_gamma(gamma)

  def _refresh_transport(self):
    """Mixture-averaged mu, lambda, D_k from Cantera. mu/lambda feed the viscous
    path (mixture law); D_k feed the Fickian species diffusion."""
    s = self.solver
    mix_visc = getattr(s, "viscous", False) and getattr(s, "_law", 0) == 2
    if not (mix_visc or self.diffusion):
      return
    rho = s.rho.cell
    e_int = (s.rhoE.cell - self._kinetic_energy()) / rho
    Y = self._mass_fractions()
    T, Pr, _ = self.chem.eos_array(rho, e_int, Y)
    mu, lam, D = self.chem.transport_array(T, Pr, Y)
    if mix_visc:
      s.set_transport(mu, lam)
    if self.diffusion:
      self._Dk = D                    # (ncells, nspec)

  def step(self, t=0.0):
    """One Strang-split reactive step; returns the hydro dt used."""
    s = self.solver
    self._refresh_transport()         # composition/T-dependent mu, lambda, D_k
    dt = s.stepper()
    self._react(0.5 * dt)             # R(dt/2)
    s.compute_fluxes(t=t)             # H(dt): hydro (convection + thermal/viscous) ...
    s.compute_new_val()
    self.species.advance(dt)          # ... + species convection
    if self.diffusion and self._Dk is not None:
      self.species.diffuse(dt, [self._Dk[:, k] for k in range(self.chem.nspec)])
    self._refresh_pressure()          # real EOS pressure after hydro
    self._react(0.5 * dt)             # R(dt/2)
    return dt
