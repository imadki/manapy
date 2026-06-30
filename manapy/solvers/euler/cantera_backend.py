#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cantera-backed real chemistry for the reactive solver (Phase 5).

Cantera is the open-source successor of CHEMKIN: it reads the same mechanism
files (chem.inp / therm.dat, via ck2yaml) and evaluates the same NASA-7
thermodynamics and Arrhenius kinetics. This adapter wraps a Cantera Solution so
manapy can use *real* multispecies thermodynamics and finite-rate chemistry
instead of the hand-fed model mechanism in `chemistry.py`/`thermo.py`.

It exposes the same conceptual interface as MixtureThermo + Chemistry:
  - thermodynamics: temperature/pressure/gamma from (rho, e_internal, Y)
  - kinetics:       net mass production rates omega_k
  - reactor:        constant-volume adiabatic 0-D integration for Strang splitting

Requires `cantera` (pip install cantera). A mechanism can be a built-in Cantera
file (e.g. 'h2o2.yaml' for H2/air) or one converted from CHEMKIN inputs:
    python -m cantera.ck2yaml --input=chem.inp --thermo=therm.dat --transport=tran.dat
"""
import numpy as np

try:
  import cantera as ct
  _HAVE_CANTERA = True
except Exception:                                   # pragma: no cover
  _HAVE_CANTERA = False


class CanteraChemistry:

  def __init__(self, mechanism="h2o2.yaml"):
    if not _HAVE_CANTERA:
      raise ImportError("cantera is required: pip install cantera")
    self.gas = ct.Solution(mechanism)
    self.mechanism = mechanism
    self.nspec = self.gas.n_species
    self.names = list(self.gas.species_names)
    self.W = self.gas.molecular_weights.copy()       # kg/kmol
    self.Ru = ct.gas_constant                        # J/kmol/K

  # --- composition helpers ---
  def index(self, name):
    return self.gas.species_index(name)

  def mass_fractions_from(self, **species):
    """Build a length-N mass-fraction vector from named species (rest = 0)."""
    Y = np.zeros(self.nspec)
    for nm, val in species.items():
      Y[self.index(nm)] = val
    s = Y.sum()
    return Y / s if s > 0 else Y

  # --- thermodynamics (single state) ---
  def _set_rhoeY(self, rho, e_int, Y):
    self.gas.Y = Y
    self.gas.UV = e_int, 1.0 / rho                   # internal energy/mass, specific volume

  def state_from_energy(self, rho, e_int, Y):
    """Return (T, P, gamma) for given density, specific internal energy and Y."""
    self._set_rhoeY(rho, e_int, Y)
    g = self.gas
    return g.T, g.P, g.cp_mass / g.cv_mass

  def gamma(self, T, P, Y):
    self.gas.TPY = T, P, Y
    return self.gas.cp_mass / self.gas.cv_mass

  def sound_speed(self, T, P, Y):
    self.gas.TPY = T, P, Y
    return self.gas.sound_speed

  # --- kinetics ---
  def production_rates(self, rho, T, Y):
    """omega_k [kg/m^3/s] from the real mechanism at (rho, T, Y)."""
    self.gas.TDY = T, rho, Y
    # net_production_rates: kmol/m^3/s -> multiply by molar mass
    return self.gas.net_production_rates * self.W

  # --- 0-D constant-volume adiabatic reactor (Strang source step) ---
  def react_cell(self, rho, T0, Y0, dt, history=False):
    """Advance one homogeneous reactor by dt; returns (T, Y) or history arrays."""
    self.gas.TDY = T0, rho, Y0
    reac = ct.IdealGasReactor(self.gas, energy="on")  # constant volume, adiabatic
    net = ct.ReactorNet([reac])
    if not history:
      net.advance(dt)
      return reac.T, reac.thermo.Y.copy()
    times = np.linspace(0.0, dt, 200)
    Ts, Ys = [], []
    for tt in times:
      net.advance(tt)
      Ts.append(reac.T)
      Ys.append(reac.thermo.Y.copy())
    return times, np.array(Ts), np.array(Ys).T

  # --- mixture-averaged transport (Cantera = open-source EGlib/CHEMKIN) ---
  def transport_array(self, T, P, Y):
    """Vectorised mixture-averaged transport from (T(n), P(n), Y(n,nspec)):
    returns viscosity mu(n), thermal conductivity lambda(n) and mass diffusion
    coefficients D(n, nspec) [m^2/s]."""
    n = T.shape[0]
    arr = ct.SolutionArray(self.gas, n)
    arr.TPY = T, P, Y
    return (arr.viscosity.copy(),
            arr.thermal_conductivity.copy(),
            arr.mix_diff_coeffs.copy())

  # --- vectorised over a whole field (n cells) via SolutionArray ---
  def eos_array(self, rho, e_int, Y):
    """Vectorised EOS: given rho(n), specific internal energy e(n) and Y(n,nspec),
    return T(n), P(n), gamma(n). Used to refresh pressure after a hydro update."""
    n = rho.shape[0]
    arr = ct.SolutionArray(self.gas, n)
    arr.UVY = e_int, 1.0 / rho, Y
    return arr.T.copy(), arr.P.copy(), (arr.cp_mass / arr.cv_mass)

  def react_array(self, rho, e_int, Y, dt, T_floor=500.0, skip_inert=True):
    """Constant-volume adiabatic reaction of a whole field over dt.

    Stiff radical chemistry needs an implicit integrator, so each cell is advanced
    with a Cantera reactor (CVODE/BDF). Cells that are essentially frozen (very
    cold, below T_floor, or with no measurable net production -- e.g. fully burnt
    or chemically inert regions) are skipped for speed. The screening threshold is
    deliberately tiny so the slow radical build-up of the induction phase is NOT
    skipped. Internal energy and density are conserved by the constant-volume
    reactor, so the hydro state (rho, momentum, rhoE) is untouched; only Y (and
    hence T, P) change. Returns (Y_new(n,nspec), T_new(n))."""
    n = rho.shape[0]
    Yc = np.array(Y, dtype=float)
    # vectorised screening: where is the chemistry active this step?
    arr = ct.SolutionArray(self.gas, n)
    arr.UVY = e_int, 1.0 / rho, Yc
    Tn = arr.T.copy()
    if skip_inert:
      wdot = arr.net_production_rates * self.W              # kg/m^3/s
      dYmax = np.max(np.abs(wdot) / rho[:, None] * dt, axis=1)
      # keep induction chemistry: only freeze truly negligible or very cold cells
      active = (dYmax > 1e-14) & (Tn > T_floor)
    else:
      active = np.ones(n, dtype=bool)

    for c in np.nonzero(active)[0]:
      self.gas.TDY = Tn[c], rho[c], Yc[c]
      reac = ct.IdealGasReactor(self.gas, energy="on")
      net = ct.ReactorNet([reac])
      try:
        net.advance(dt)
        Yc[c] = reac.thermo.Y
        Tn[c] = reac.T
      except Exception:
        pass                                               # keep frozen on failure
    return Yc, Tn

  def equilibrium_T(self, rho, T0, Y0):
    """Constant-UV equilibrium temperature (adiabatic flame T at constant volume)."""
    self.gas.TDY = T0, rho, Y0
    u0 = self.gas.int_energy_mass
    self.gas.equilibrate("UV")
    return self.gas.T
