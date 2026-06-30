# Compressible reactive-flow physics for manapy (`solvers/euler`)

This documents the high-fidelity compressible-flow physics added on top of the
cell-centred unstructured finite-volume `EulerSolver`. Each capability is built as
a composable module and validated against analytic / reference solutions.

Legend: ✅ done · ⬜ todo

## Core (`system.py`, `fvm_utils{2,3}d_compute.py`)
- ✅ Compressible Euler/NS, Rusanov + Roe fluxes, MUSCL order 2 (2D)
- ✅ Viscous NS (Newtonian stress + Fourier conduction), constant/Sutherland — 2D & 3D
- ✅ Non-reflecting characteristic far-field BC (NSCBC, Riemann-invariant) — 2D & 3D
- Validated: **exact Sod Riemann solution** (Roe < Rusanov), viscous decay (analytic).

## LES / subgrid-scale turbulence
- ✅ Smagorinsky and WALE eddy viscosity (2D/3D), added to the viscous stress with a
  turbulent Prandtl number. Example `examples/2D/les_taylorgreen2d.py`.

## Turbulence diagnostics (`diagnostics.py`)
- ✅ vorticity, enstrophy, Q-criterion, resolved TKE (2D/3D). Validated on solid-body
  rotation (omega=2Omega, Q=Omega^2) and pure shear (Q=0).

## Multispecies transport (`species.py`, `species_compute.py`)
- ✅ N species partial densities, Rusanov convection consistent with the bulk flux
  (ΣY=1, exact species-mass conservation, advection at the flow speed).
- ✅ Fickian diffusion ρ·D_k·∇Y_k (validated vs analytic Gaussian spreading, 2·D·t).

## Thermodynamics (`thermo.py`)
- ✅ variable-gamma mixture of calorically-perfect gases (`MixtureThermo`): R, cp, cv,
  gamma(Y), EOS T(rhoE)/P, formation energy e0. Validated: air props, EOS round-trip.

## Reactive chemistry (`chemistry.py`, `cantera_backend.py`, `reactive_solver.py`)
- ✅ model engine: Arrhenius mass-action kinetics + constant-UV 0-D reactor.
- ✅ **real chemistry via Cantera** (open-source CHEMKIN/EGlib): thermo, kinetics and
  mixture-averaged transport (μ, λ, D_k). Validated H2/air ignition (T→UV equilibrium).
- ✅ `ReactiveSolver`: Strang operator splitting (hydro + species + per-cell reaction),
  real-EOS pressure feedback. Validated on a constant-volume bomb.

## Turbulent inflow (`inflow.py`)
- ✅ synthetic-turbulence shear-layer inlet: tanh mean profile + white-noise
  fluctuations at a target intensity (optional digital-filter correlation).

## High-order WENO on unstructured meshes (`weno.py`)
Following Tsoutsanis, J. Comput. Phys. 475 (2023) 108840.
- ✅ Milestone 1 — k-exact least-squares reconstruction (central vertex-based stencil,
  exact cell moments, precomputed pseudo-inverse). Validated to 2.5e-13 k-exactness.
- ⬜ smoothness indicators + nonlinear WENO weights; directional stencils; edge-quadrature
  flux + solver coupling.

## Variable-gamma (multispecies) hydro coupling
- ✅ per-cell ratio of specific heats in the Rusanov wave speed and the pressure
  update (`*_rusanov_vg`, `_update_euler_2d_vg`; `EulerSolver(variable_gamma=True)` +
  `set_gamma`). Wired into `ReactiveSolver` (gamma tracks composition each step).
  Validated: exact match to the scalar path for uniform gamma; ~15x lower spurious
  pressure error at a moving multi-gamma contact (2D rusanov order 1).
- ⬜ double-flux / quasi-conservative variant to remove the residual contact
  pressure oscillation; 3D and Roe variable-gamma paths.

## Per-boundary BC dispatch
- ✅ `EulerSolver(bc={name: type})` applies a different treatment per named boundary
  (`neumann`, `slipwall`, `nonreflecting`) in one run (`_apply_per_boundary_ghosts`).
  Validated: all-neumann map matches the scalar `Neumann` exactly; a uniform channel
  stream with slip walls + non-reflecting in/out stays steady to 1e-18; an acoustic
  pulse radiates out axially (8.9x less trapped than all-walls). Example
  `examples/2D/mixed_bc_channel2d.py`. (2D; inflow combines via `TurbulentInflow.apply`.)

## Mixture transport in the viscous path
- ✅ `viscosity_law="mixture"`: per-cell μ, λ supplied each step (face-averaged into
  the laminar base via `EulerSolver.set_transport`), wired in `ReactiveSolver` from
  Cantera `transport_array`. Validated: uniform-μ mixture matches the constant law
  exactly; in a reacting bomb μ rises x1.94 and λ x1.71 as T goes 1100->2928 K.
  Composes with the LES SGS (μ_t added on top of the mixture base).

## Double-flux method (Abgrall–Billet) — multi-gamma pressure equilibrium
- ✅ `EulerSolver(variable_gamma=True, doubleflux=True)`: each cell is updated with its
  own frozen gamma, the neighbour energy reinterpreted in that frame, and the conserved
  energy re-synced from pressure each step (`_doubleflux_residual_euler_2d`,
  `_update_euler_2d_df`). Removes the spurious contact pressure oscillation.
  Validated: a moving multi-gamma contact keeps P to **machine precision** (2.9e-15 vs
  2.8e-2 for variable-gamma alone); a strong flame-like contact (density ratio ~7,
  gamma 1.25/1.40) stays exact (2.3e-15) and stable; uniform gamma reproduces the
  standard scheme (stable Sod). This unblocks the multi-component flame stability.

## Propagating 1-D flame (capstone) — core unblocked
The full stack is wired and validated piece-by-piece (convection, diffusion, chemistry,
mixture transport, variable-gamma, and now the double-flux that keeps the burnt/unburnt
contact in pressure equilibrium). Remaining to a converged propagating flame:
- ⬜ in `ReactiveSolver`, carry the **sensible** energy in the hydro (so the double-flux
  re-sync preserves the chemical/formation energy carried by the advected species),
  and add the heat release back in the reaction step.
- ⬜ flame-speed validation vs Cantera `FreeFlame` (S_L ≈ 2.56 m/s, stoich H2/air);
  needs a flame-resolving mesh (HPC-scale steps for the low-Mach acoustic limit).

## Cross-cutting refinements (optional)
- ⬜ correction velocity for exact sum conservation with unequal D_k
- ⬜ 3D / Roe variable-gamma; per-boundary dispatch in 3D
