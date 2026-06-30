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

  def __init__(self, solver, chemistry, Y0, diffusion=False, sensible_energy=None):
    """
    solver    : EulerSolver (hydro; pass a representative constant gamma)
    chemistry : CanteraChemistry (real thermo + kinetics)
    Y0        : initial mass fractions, list/array length chem.nspec (scalar or
                per-cell), ordered as chemistry.names.
    diffusion : if True, add Fickian species diffusion (mixture-averaged D_k from
                Cantera) each step -- required for propagating premixed flames,
                where heat and radicals diffuse into the unburnt gas.
    sensible_energy : carry the *sensible* energy in the hydro and the chemical
                (formation) energy in the advected species, injecting the reaction
                heat release into the sensible energy each react sub-step. Required
                for the double-flux update (which re-syncs rhoE = P/(gamma-1)+KE,
                a sensible relation) to stay consistent across a flame; otherwise
                the formation energy would be silently dropped at the burnt/unburnt
                contact. Defaults to the solver's double-flux setting. The conserved
                physical energy is `total_energy()`; rho.rhoE holds the sensible part.
    """
    self.solver = solver
    self.chem = chemistry
    self.dim = solver.dim
    self.diffusion = bool(diffusion)
    if len(Y0) != chemistry.nspec:
      raise ValueError("Y0 length must equal the number of mechanism species")
    if sensible_energy is None:
      sensible_energy = bool(getattr(solver, "_doubleflux", False))
    self.sensible = bool(sensible_energy)
    # species transport carries the partial densities, in mechanism order
    self.species = SpeciesTransport(solver, Y0, names=chemistry.names, renormalize=True)
    self._Dk = None
    if self.sensible:
      # The double-flux update re-syncs rhoE = P/(gamma-1)+KE, which is a *sensible*
      # (zero-formation) energy. The user initialises rhoE with the total internal
      # energy; set P/gamma from it, then convert rhoE to the sensible form the
      # hydro will carry from here on. The true temperature is afterwards always
      # recovered exactly from the pressure (T = P/(rho R), ideal-gas law), so the
      # chemistry never sees the truncated hydro energy.
      s = self.solver
      e_total = (s.rhoE.cell - self._kinetic_energy()) / s.rho.cell
      _, P, gamma = self.chem.eos_array(s.rho.cell, e_total, self._mass_fractions())
      s.P.cell[:] = P
      if getattr(s, "variable_gamma", False):
        s.set_gamma(gamma)
      s.rhoE.cell[:] = P / (gamma - 1.0) + self._kinetic_energy()
    else:
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

  def _temperature(self):
    """Cell temperature. In the sensible (double-flux) mode the hydro energy is the
    truncated P/(gamma-1) form, so the temperature is recovered exactly from the
    real-EOS pressure via the ideal-gas law T = P/(rho R(Y)); otherwise it comes
    from the total internal energy carried in rhoE."""
    s = self.solver
    Y = self._mass_fractions()
    if self.sensible:
      return s.P.cell / (s.rho.cell * self.chem.Rspecific(Y))
    e_int = (s.rhoE.cell - self._kinetic_energy()) / s.rho.cell
    T, _, _ = self.chem.eos_array(s.rho.cell, e_int, Y)
    return T

  def _internal_energy(self):
    """Specific *total* internal energy (Cantera reference, incl. formation). In
    the sensible mode it is reconstructed exactly from the pressure-recovered
    temperature; otherwise it is the energy carried in rhoE."""
    s = self.solver
    if self.sensible:
      return self.chem.internal_energy_array(self._temperature(), self._mass_fractions())
    return (s.rhoE.cell - self._kinetic_energy()) / s.rho.cell

  def total_energy(self):
    """Conserved physical total energy density rho*u(T,Y) + KE. With the sensible
    split, s.rhoE.cell holds only the truncated sensible+kinetic part; the chemical
    (formation) energy lives in the composition and is restored here."""
    s = self.solver
    return s.rho.cell * self._internal_energy() + self._kinetic_energy()

  def _refresh_pressure(self):
    """Refresh P and the per-cell gamma after a hydro step.

    Default (total-energy) mode: P and gamma from the real EOS on rhoE. Sensible
    (double-flux) mode: the conservative double-flux update already wrote a
    pressure-equilibrium-preserving P = (gamma-1)(rhoE-KE); we keep it and only
    re-evaluate gamma for the advected composition/temperature (the next step's
    frozen value)."""
    s = self.solver
    Y = self._mass_fractions()
    if self.sensible:
      if getattr(s, "variable_gamma", False):
        s.set_gamma(self.chem.gamma_array(self._temperature(), Y))
      return
    e_int = (s.rhoE.cell - self._kinetic_energy()) / s.rho.cell
    _, P, gamma = self.chem.eos_array(s.rho.cell, e_int, Y)
    s.P.cell[:] = P
    if getattr(s, "variable_gamma", False):
      s.set_gamma(gamma)

  def _react(self, dt_r):
    """Constant-volume reaction sub-step over dt_r; updates Y, T, P.

    The reactor conserves the *total* internal energy e_int (formation + sensible)
    at fixed volume; only the composition changes. In the sensible (double-flux)
    mode rhoE is then re-synced to the new pressure (rhoE = P/(gamma-1)+KE): as the
    reaction releases formation energy the temperature -- and hence P -- rises, so
    rhoE rises by exactly the combustion heat release. Without the split rhoE is the
    total energy and stays fixed (the reactor conserves it)."""
    s = self.solver
    rho = s.rho.cell
    e_int = self._internal_energy()                 # total internal energy (conserved)
    Y = self._mass_fractions()
    Ynew, _ = self.chem.react_array(rho, e_int, Y, dt_r)
    for k, qk in enumerate(self.species.q):
      qk.cell[:] = rho * Ynew[:, k]
    # e_int (total) is conserved by the reactor; new P, gamma from the new Y
    _, P, gamma = self.chem.eos_array(rho, e_int, Ynew)
    s.P.cell[:] = P
    if getattr(s, "variable_gamma", False):
      s.set_gamma(gamma)
    if self.sensible:
      # deposit the heat release into the hydro (sensible) energy via the new P
      s.rhoE.cell[:] = P / (gamma - 1.0) + self._kinetic_energy()

  def _refresh_transport(self):
    """Mixture-averaged mu, lambda, D_k from Cantera. mu/lambda feed the viscous
    path (mixture law); D_k feed the Fickian species diffusion."""
    s = self.solver
    mix_visc = getattr(s, "viscous", False) and getattr(s, "_law", 0) == 2
    if not (mix_visc or self.diffusion):
      return
    rho = s.rho.cell
    e_int = self._internal_energy()
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
