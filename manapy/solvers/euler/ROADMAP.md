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

## Cross-cutting refinements (optional)
- ⬜ per-boundary BC dispatch (mixed inflow / non-reflecting outflow / walls in one run)
- ⬜ variable-gamma in the hydro fluxes; μ/λ from the mixture transport into the viscous path
- ⬜ correction velocity for exact sum conservation with unequal D_k; propagating 1-D flame
