# Compressible reactive-flow physics for manapy (`solvers/euler`)

This documents the high-fidelity compressible-flow physics added on top of the
cell-centred unstructured finite-volume `EulerSolver`. Each capability is built as
a composable module and validated against analytic / reference solutions.

Legend: ✅ done · ⬜ todo

## Core (`system.py`, `fvm_utils{2,3}d_compute.py`)
- ✅ Compressible Euler/NS, Rusanov + Roe fluxes, MUSCL order 2 (2D & 3D). 3D MUSCL
  validated positive/non-oscillatory and more accurate than order 1 vs exact Sod
  (`test_muscl_order2_3d_sharper_than_order1`).
- ✅ Viscous NS (Newtonian stress + Fourier conduction), constant/Sutherland — 2D & 3D
- ✅ Non-reflecting characteristic far-field BC (NSCBC, Riemann-invariant) — 2D & 3D
- ✅ Roe scheme with Harten entropy fix (`entropy_fix`) on the acoustic waves — 2D & 3D.
  Validated: ef=0 reproduces plain Roe; ef>0 stays positive and only smooths the
  sonic point locally (`test_roe_entropy_fix{,_3d}`).
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
- ✅ k-exact least-squares reconstruction (central vertex-based stencil, exact cell
  moments, precomputed pseudo-inverse) -- k-exactness to 2.5e-13.
- ✅ oscillation matrix OI + smoothness indicator SI=a^T OI a -- ~0 for smooth data,
  blows up (x1500) at discontinuity cells.
- ✅ directional (sectoral) stencils from the two-ring node-neighbour pool.
- ✅ non-linear WENO weighting (`weno_reconstruct`): k-exact in smooth regions
  (1.6e-11) and non-oscillatory at shocks (Gibbs overshoot 0.65 -> 0).
- ✅ per-step reconstruction compiled with **numba** over the precomputed packed
  arrays (~81x faster than the Python loop).
- ✅ flux coupling on scalar linear advection (`advect_residual`, upwind WENO face
  value + SSP-RK3): square wave non-oscillatory, Gaussian peak ~0.95 vs ~0.58 for
  first order. Example `examples/2D/weno_advection2d.py`.
- ✅ **Euler system coupling** (`weno_euler.py::WenoEulerSolver`): per-variable WENO
  reconstruction + Rusanov Riemann flux at face centres + SSP-RK3, numba kernel.
  Validated on Sod vs exact Riemann: L2 1.78e-2 vs 3.80e-2 for first order (53% better).
  Example `examples/2D/weno_euler_sod2d.py`.
- ✅ **Shu-Osher** (Mach-3 shock into a sin density field) on the bundled square mesh
  (uniform in y, fine in x): WENO keeps the entrained post-shock oscillation
  (amplitude 0.142) that first-order Rusanov diffuses away (0.061) — 2.3x more
  resolved structure. Example `examples/2D/weno_euler_shuosher2d.py`.
- ✅ **3D unstructured WENO reconstruction** (`WenoReconstruction` is now
  dimension-generic; 3D kernels `_weno_cm_3d` / `_weno_select_3d` / `_weno_build_3d`).
  Polyhedral cell moments via a centroid-apex tetra decomposition of each face
  (simplex moment formula); axis-directional two-ring stencils; SVD build with
  near-null truncation so sliver/boundary stencils stay bounded. Validated on the
  3D mesh: zeroth moment = cell volume (2e-13); k-exact on a quadratic to machine
  precision on the well-conditioned bulk (median ~9e-16, p90 ~2e-13) with a bounded
  pseudo-inverse everywhere (|pinv| 6.8e12 -> 30); smoothness indicator x59 at a
  discontinuity; WENO weighting non-oscillatory (overshoot 0 vs 1.17 for the plain
  linear fit). Tests `tests/test_weno3d.py`. (The hot reconstruction loop is shared
  with 2D.)
- ✅ **3D WENO Euler system coupling** (`WenoEulerSolver` is now dimension-generic;
  numba kernel `_weno_euler_rusanov_3d`): per-variable WENO reconstruction of
  (rho, rho*u, rho*v, rho*w, rho*E) + Rusanov flux at face centres + SSP-RK3.
  Validated on a 3D Sod: positive (rho, P > 0), essentially non-oscillatory (rho in
  [0.125, 1] up to a ~1e-4 WENO overshoot). Test `test_weno3d_euler_sod_...`; example
  `examples/3D/weno_euler_sod3d.py`. 2D coupling unchanged (Sod L2 1.78e-2).
- ✅ **MPI / distributed WENO**: the reconstruction stencils span partition
  boundaries via the halo (`cells.halonid`). At build time the halo cells are
  appended to an extended `[local|halo]` geometry (positions/volumes from
  `halos.centvol`, central moments exchanged once); the stencil pool adds the halo
  node-neighbours (encoded `nb+halo_id`) and the SVD build truncates near-null
  directions (`1e-10`) so partial-halo boundary stencils stay bounded. Each RK
  stage exchanges the field to the halo (`_extend_field`) before reconstruction.
  `WenoEulerSolver` takes the halo neighbour state at partition faces (name==10),
  so the interior is full WENO and only the one-cell partition interface is first
  order. Validated: build no longer crashes under MPI; a 2-rank Sod matches the
  serial run (rho min identical, max/mass to ~5 digits); near-linear speedup
  (453→223→105 ms/step at 1/2/4 ranks on 92k cells).
- ⬜ edge Gauss quadrature for curved/large cells; characteristic-variable
  reconstruction; full-order (exchanged-coefficient) flux at partition faces.

## Variable-gamma (multispecies) hydro coupling
- ✅ per-cell ratio of specific heats in the Rusanov wave speed and the pressure
  update (`*_rusanov_vg`, `_update_euler_2d_vg`; `EulerSolver(variable_gamma=True)` +
  `set_gamma`). Wired into `ReactiveSolver` (gamma tracks composition each step).
  Validated: exact match to the scalar path for uniform gamma; ~15x lower spurious
  pressure error at a moving multi-gamma contact (2D rusanov order 1).
- ✅ **3D variable-gamma + double-flux** (`fvm_utils3d`: `*_rusanov_vg`,
  `doubleflux_residual_euler_3d`, `update_euler_3d_{vg,df}`; `EulerSolver` now wires
  rusanov order-1 variable_gamma/doubleflux in 2D **and** 3D). The `ReactiveSolver`
  (dimension-agnostic) + sensible-energy split therefore run in 3D unchanged.
  Validated: a moving 3D multi-gamma contact keeps P to **machine precision**
  (L2 1.6e-14 vs 1.2e-2 for variable-gamma alone), and a 3D reactive double-flux
  bomb conserves the physical total energy to 4.7e-15. Test
  `test_doubleflux_contact_pressure_equilibrium_3d`; example
  `examples/3D/multigamma_contact3d.py`.
- ⬜ Roe variable-gamma path (2D & 3D currently rusanov-only).

## Per-boundary BC dispatch
- ✅ `EulerSolver(bc={name: type})` applies a different treatment per named boundary
  (`neumann`, `slipwall`, `nonreflecting`) in one run (`_apply_per_boundary_ghosts`).
  Validated: all-neumann map matches the scalar `Neumann` exactly; a uniform channel
  stream with slip walls + non-reflecting in/out stays steady to 1e-18; an acoustic
  pulse radiates out axially (8.9x less trapped than all-walls). Example
  `examples/2D/mixed_bc_channel2d.py`. (inflow combines via `TurbulentInflow.apply`.)
- ✅ **3D** per-boundary dispatch (`_apply_per_boundary_ghosts_3d`, adds the
  z-momentum/w-velocity to neumann/slipwall/nonreflecting). Validated: a uniform 3D
  stream with slip lateral walls + non-reflecting in/out stays steady to ~1e-13
  (`test_per_boundary_dispatch_3d`).

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
- ✅ **sensible-energy split** in `ReactiveSolver(sensible_energy=...)`, auto-on under
  double-flux. The hydro carries only the sensible energy (rhoE = P/(gamma-1)+KE, the
  form the double-flux update re-syncs); the chemical/formation energy lives in the
  composition. Temperature is recovered exactly from the pressure (T = P/(rho R(Y)),
  ideal-gas law — no calorically-perfect error), the reaction runs through Cantera's
  exact constant-UV reactor, and the heat release flows back into rhoE via the new
  pressure. Validated: a quiescent H2/air bomb reaches the same constant-UV equilibrium
  T (2928 K) as the total-energy coupling while conserving the *physical* total energy
  to machine precision (dE/E ~ 2.5e-15). Test `test_reactive_sensible_energy_double_flux`.
- ⬜ flame-speed validation vs Cantera `FreeFlame` (S_L ≈ 2.56 m/s, stoich H2/air);
  needs a flame-resolving mesh (HPC-scale steps for the low-Mach acoustic limit).

## Cross-cutting refinements (optional)
- ⬜ correction velocity for exact sum conservation with unequal D_k
- ⬜ Roe variable-gamma (rusanov-only for now)
