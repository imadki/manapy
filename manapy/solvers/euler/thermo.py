#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multispecies thermodynamics: variable-gamma mixture of calorically-perfect gases.

This is the composition-dependent equation of state underpinning the reactive
solver (Phase 4). Each species k is a calorically-perfect gas with
constant specific heat cp_k and specific gas constant R_k = R_universal / W_k.
Mixture properties are mass-fraction weighted:

    R_mix   = sum_k Y_k R_k
    cp_mix  = sum_k Y_k cp_k
    cv_mix  = cp_mix - R_mix
    gamma   = cp_mix / cv_mix              (varies in space with composition)

Calorically-perfect EOS (internal energy referenced to T=0):
    e_int = cv_mix * T
    P     = rho * R_mix * T = (gamma-1) * rho * e_int
    T     = e_int / cv_mix
    c     = sqrt(gamma * P / rho)

A single species reduces exactly to the constant-gamma ideal gas. NASA-7
thermally-perfect polynomials (cp_k(T), h_k(T) + Newton for T) are a later
refinement (Phase 4b) that slots in behind the same interface.
"""
import numpy as np

R_UNIVERSAL = 8314.462618   # J / (kmol K)


class MixtureThermo:

  def __init__(self, W, cp=None, gamma=None, e0=None, names=None, R_universal=R_UNIVERSAL):
    """
    W     : per-species molar mass [kg/kmol]
    cp    : per-species specific heat at constant pressure [J/(kg K)]  (optional)
    gamma : per-species ratio of specific heats (optional alternative to cp)
    e0    : per-species formation internal energy [J/kg] referenced to T=0
            (optional; defaults to 0 -> inert calorically-perfect gas). The
            chemical energy sum_k Y_k e0_k is what drives combustion heat release.
    Provide exactly one of cp or gamma. With gamma, cp_k = gamma_k R_k/(gamma_k-1).
    """
    self.W = np.asarray(W, dtype=float)
    self.nspec = self.W.size
    self.Ru = float(R_universal)
    self.R = self.Ru / self.W                       # specific gas constant per species
    if (cp is None) == (gamma is None):
      raise ValueError("provide exactly one of cp or gamma")
    if cp is not None:
      self.cp = np.asarray(cp, dtype=float)
    else:
      g = np.asarray(gamma, dtype=float)
      self.cp = g * self.R / (g - 1.0)
    self.cv = self.cp - self.R
    self.gamma_k = self.cp / self.cv
    if np.any(self.cv <= 0.0):
      raise ValueError("non-physical species: cv = cp - R must be > 0")
    self.e0 = np.zeros(self.nspec) if e0 is None else np.asarray(e0, dtype=float)
    self.names = names if names is not None else [f"sp{k}" for k in range(self.nspec)]

  def chemical_energy(self, Y):
    """Specific chemical (formation) internal energy sum_k Y_k e0_k."""
    out = 0.0
    for k in range(self.nspec):
      out = out + Y[k] * self.e0[k]
    return out

  # --- mixture properties from mass fractions (each Y[k] is a scalar or array) ---
  def mixture_R(self, Y):
    out = 0.0
    for k in range(self.nspec):
      out = out + Y[k] * self.R[k]
    return out

  def mixture_cp(self, Y):
    out = 0.0
    for k in range(self.nspec):
      out = out + Y[k] * self.cp[k]
    return out

  def mixture_cv(self, Y):
    out = 0.0
    for k in range(self.nspec):
      out = out + Y[k] * self.cv[k]
    return out

  def mixture_gamma(self, Y):
    return self.mixture_cp(Y) / self.mixture_cv(Y)

  def mixture_W(self, Y):
    """Mixture molar mass: 1/W = sum_k Y_k / W_k."""
    inv = 0.0
    for k in range(self.nspec):
      inv = inv + Y[k] / self.W[k]
    return 1.0 / inv

  # --- equation of state ---
  def temperature(self, rho, rhoE, ke, Y):
    """T from total energy, removing kinetic and chemical parts:
       T = (rhoE - ke - rho*chem) / (rho*cv_mix).   ke = 0.5 rho |u|^2."""
    return (rhoE - ke - rho * self.chemical_energy(Y)) / (rho * self.mixture_cv(Y))

  def pressure_from_T(self, rho, T, Y):
    return rho * self.mixture_R(Y) * T

  def pressure_from_E(self, rho, rhoE, ke, Y):
    """Ideal-gas pressure from the sensible temperature: P = rho R_mix T(rhoE).
    With e0=0 this reduces to (gamma_mix-1)(rhoE-ke)."""
    return self.pressure_from_T(rho, self.temperature(rho, rhoE, ke, Y), Y)

  def internal_energy(self, T, Y):
    """Total specific internal energy e = cv_mix T + chemical."""
    return self.mixture_cv(Y) * T + self.chemical_energy(Y)

  def total_energy(self, rho, T, Y, ke):
    """rhoE = rho (cv_mix T + chemical) + ke."""
    return rho * (self.mixture_cv(Y) * T + self.chemical_energy(Y)) + ke

  def sound_speed(self, P, rho, Y):
    return np.sqrt(self.mixture_gamma(Y) * P / rho)
