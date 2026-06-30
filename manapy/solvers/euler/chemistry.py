#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-rate chemistry: Arrhenius mass-action kinetics + a 0-D reactor.

The reactive chemistry core (Phase 5). A mechanism is a set of reactions

    sum_k nu'_kj X_k  ->  sum_k nu''_kj X_k ,   j = 1..M

with forward rate constant k_fj(T) = A_j T^beta_j exp(-Ea_j / (Ru T)) and
mass-action rate of progress q_j = k_fj * prod_k [X_k]^nu'_kj (irreversible form;
reversible mechanisms add the backward term via an equilibrium constant later).
Molar concentrations are [X_k] = rho Y_k / W_k. The species mass production rate is

    omega_k = W_k * sum_j (nu''_kj - nu'_kj) q_j      [kg / m^3 / s].

Heat release is carried implicitly by the species formation energies e0_k in
MixtureThermo: a constant-volume adiabatic reactor conserves total internal
energy, so converting reactants (high e0) to products (low e0) raises the
temperature -> ignition.

This validates the kinetics independently of the hydro solver; Strang splitting
then couples `react_cell` per cell into the time loop.
"""
import numpy as np

from manapy.solvers.euler.thermo import R_UNIVERSAL


class Chemistry:

  def __init__(self, thermo, nu_reac, nu_prod, A, beta, Ea):
    """
    thermo  : MixtureThermo (provides W, cv, e0, ...)
    nu_reac : (M, N) reactant stoichiometric coefficients (molar)
    nu_prod : (M, N) product stoichiometric coefficients
    A,beta,Ea: (M,) Arrhenius parameters. Units of A are mechanism-dependent
              (consistent with [X] in kmol/m^3 and the reaction order); Ea in J/kmol.
    """
    self.thermo = thermo
    self.W = thermo.W
    self.nspec = thermo.nspec
    self.nu_reac = np.asarray(nu_reac, dtype=float).reshape(-1, self.nspec)
    self.nu_prod = np.asarray(nu_prod, dtype=float).reshape(-1, self.nspec)
    self.nu = self.nu_prod - self.nu_reac           # net stoichiometry (M, N)
    self.nreac = self.nu.shape[0]
    self.A = np.atleast_1d(np.asarray(A, dtype=float))
    self.beta = np.atleast_1d(np.asarray(beta, dtype=float))
    self.Ea = np.atleast_1d(np.asarray(Ea, dtype=float))
    self.Ru = R_UNIVERSAL
    # mass-conservation sanity: sum_k W_k nu_kj should be ~0 for every reaction
    massbal = self.nu @ self.W
    if np.any(np.abs(massbal) > 1e-6 * self.W.max()):
      raise ValueError(f"reactions not mass-conserving: sum_k W_k nu_kj = {massbal}")

  def production_rates(self, rho, T, Y):
    """omega_k [kg/m^3/s] for a single state (Y is a length-N sequence of scalars)."""
    Yv = np.asarray([Y[k] for k in range(self.nspec)], dtype=float)
    conc = rho * Yv / self.W                        # [X_k] in kmol/m^3
    conc = np.maximum(conc, 0.0)
    omega = np.zeros(self.nspec)
    for j in range(self.nreac):
      kf = self.A[j] * T ** self.beta[j] * np.exp(-self.Ea[j] / (self.Ru * T))
      q = kf
      for k in range(self.nspec):
        p = self.nu_reac[j, k]
        if p != 0.0:
          q *= conc[k] ** p
      omega += self.W * self.nu[j] * q
    return omega

  def react_cell(self, rho, T0, Y0, t_end, nsub=None, history=False):
    """Constant-volume adiabatic 0-D reactor over [0, t_end].

    Returns (T, Y) by default, or (t_arr, T_arr, Y_arr) if history=True.
    Total specific internal energy is conserved, so T follows from the
    composition each evaluation. Uses scipy BDF (stiff); falls back to explicit
    substepping if scipy is unavailable.
    """
    thd = self.thermo
    Y0 = np.asarray(Y0, dtype=float)
    e_const = thd.mixture_cv(Y0) * T0 + thd.chemical_energy(Y0)

    def temp(Yv):
      Yl = [Yv[k] for k in range(self.nspec)]
      return (e_const - thd.chemical_energy(Yl)) / thd.mixture_cv(Yl)

    def rhs(t, Yv):
      Yc = np.maximum(Yv, 0.0)
      T = temp(Yc)
      omega = self.production_rates(rho, T, [Yc[k] for k in range(self.nspec)])
      return omega / rho

    try:
      from scipy.integrate import solve_ivp
      teval = np.linspace(0.0, t_end, 200) if history else None
      sol = solve_ivp(rhs, (0.0, t_end), Y0, method="BDF",
                      rtol=1e-8, atol=1e-12, t_eval=teval)
      if history:
        Yh = np.maximum(sol.y, 0.0)
        Yh = Yh / Yh.sum(axis=0, keepdims=True)
        Th = np.array([temp(Yh[:, i]) for i in range(Yh.shape[1])])
        return sol.t, Th, Yh
      Yend = np.maximum(sol.y[:, -1], 0.0)
    except Exception:
      Yend = Y0.copy()
      nn = nsub if nsub is not None else 100000
      dt = t_end / nn
      ts, Ts, Ys = [0.0], [T0], [Y0.copy()]
      for i in range(nn):
        Yend = np.maximum(Yend + dt * rhs(0.0, Yend), 0.0)
        if history and (i % max(1, nn // 200) == 0):
          Yn = Yend / Yend.sum()
          ts.append((i + 1) * dt); Ts.append(temp(Yn)); Ys.append(Yn)
      if history:
        return np.array(ts), np.array(Ts), np.array(Ys).T
    Yend = Yend / Yend.sum()
    return temp(Yend), Yend
