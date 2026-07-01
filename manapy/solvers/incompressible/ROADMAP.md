# Incompressible Navier-Stokes (`solvers/incompressible`)

Incompressible laminar transient flow on the unstructured collocated finite-volume
grid -- the manapy analogue of OpenFOAM's `icoFoam`. Phase 1 of the multiphase
(interFoam-style VOF) roadmap: the single-phase pressure-velocity coupling is the
prerequisite for the two-phase solver.

Legend: ✅ done & validated · 🟡 done, refinement pending · ⬜ todo

## Method (`system.py`, `fvm_utils_compute.py`)
- ✅ **face-flux-consistent Chorin projection**. Per step:
  1. predictor `u* = u + dt(-conv + nu*diff)` (convection by the divergence-free face
     flux, first-order upwind; two-point face diffusion)
  2. face flux `phi* = u*_face . S_f`
  3. pressure `A P = -(rho/dt) sum_f phi*_f` (two-point Laplacian `a_f = area/dist`,
     one Dirichlet reference to remove the pure-Neumann singularity)
  4. correct `phi = phi* - (dt/rho) a_f (P_N - P_P)` (divergence-free **by construction**)
     and `u = u* - (dt/rho) grad P` (Green-Gauss cell reconstruction)
- The divergence, the pressure Laplacian and the correction share the **same** face
  coefficient `a_f`, so the corrected face flux is exactly divergence-free -- this is
  what makes the collocated method stable (a mismatched div/grad/Laplacian triple
  blows up, as an earlier diamond-Laplacian + Green-Gauss attempt did).

## Validation
- ✅ **lid-driven cavity vs Ghia, Ghia & Shin (1982), Re=100**: reaches steady state,
  u on the vertical centreline matches the benchmark (u_min = -0.222 vs -0.211;
  profile L2 ~ 0.02 on a coarse ~3.7k-cell mesh). Example
  `examples/2D/lid_driven_cavity2d.py`.

## Known limitations / next steps
- ✅ the pressure Poisson **reuses manapy's `LinearSolver(scheme='fv')`** (the two-point
  cell Laplacian, `fv_coeff = |Sf|^2/|Sf.d|`, halo/MPI coupling built in) -- no bespoke
  matrix, and the **backend is a free choice** (PETSc / MUMPS / Ginkgo), like the darcy
  example. (`scheme='diamond'` is the wider vertex-based Laplacian; it needs its own
  face-flux correction, `compute_face_gradient`, to stay consistent -- a future option.)
  Only the collocated-specific pieces (momentum convection by the divergence-free face
  flux, the cell divergence) are new numba kernels, since manapy has no such operator.
- ✅ **MPI**: the pressure Poisson is distributed (fv scheme couples the halos in the
  matrix); the momentum/divergence/gradient kernels exchange the velocity/pressure to
  the halo and treat partition faces (name==10) with the neighbour-rank value. Validated:
  a 2-rank cavity matches the serial run to ~1% on a transient (converges at steady),
  stable and finite.
- ✅ **PISO-style pressure correctors** (`ncorr`, default 2): the predictor is followed
  by a loop of {divergence -> pressure solve -> velocity correction}; iterating drives
  the residual collocated cell divergence down (0.20 -> 0.13 -> 0.10 at nCorr 1/2/4 on
  the cavity) while the steady solution is unchanged. This is the PISO outer loop.
  🟡 the momentum predictor is still **explicit**, so the correctors are an iterated
  projection rather than icoFoam's implicit-momentum PISO (a_P/H split + Rhie-Chow);
  an implicit momentum solve is the remaining step to faithful icoFoam.
- ⬜ higher-order convection (currently first-order upwind); non-orthogonal correction
  in the two-point face gradient (currently the orthogonal approximation).
- ⬜ 3D.

## True PISO (implicit momentum) — implementation plan for the next session
The current solver is already PISO with `a_P = rho*V/dt` (explicit momentum). "True"
PISO makes the momentum implicit -> larger stable `dt` (implicit convection) and
faithfulness to icoFoam. Inventory already done (do NOT re-investigate):
- manapy has **no** implicit convection/momentum matrix assembly (only explicit
  `advecdiff`) and **no** variable-coefficient Laplacian (the `fv` scheme uses a fixed
  `faces.fv_coeff`). All LS backends expose `set_matrix(row,col,data)` + `with_mtx=True`
  (0-based global indices via `cells.loctoglob`; halo column = `halos.halosext[h,0]`).

Steps:
1. **Momentum matrix** `M = (rho*V/dt) I + C(implicit upwind conv) + nu*L_fv`, assembled
   as global triplets (reuse `fv_coeff` for the diffusion two-point part; upwind conv
   from the current face flux). Start **semi-implicit** (implicit diffusion, deferred
   convection) to de-risk, then add implicit convection. Solve u,v via a chosen manapy
   LS (`set_matrix`, backend a free choice). Extract `a_P` = diagonal of M.
2. **Predictor**: solve `M u = (rho*V/dt) u^n - grad(p^n) [+ deferred conv]`.
3. **PISO correctors** (`ncorr`): pseudo-velocity `H/a_P` (H = off-diagonal action +
   sources); pressure eq `div( (1/a_P)_face grad p ) = div( (H/a_P)_face )` -- a
   **variable-coefficient** Laplacian = `(1/a_P)_face * fv_coeff`, so assemble it each
   step via `set_matrix` (matrix changes as `a_P` changes; no `reuse_mtx`). Correct the
   face flux `phi -= (1/a_P)_face (grad p . S)` and the cell velocity `u = H/a_P -
   (1/a_P) grad p`.
4. **Validate** vs OpenFOAM `icoFoam` (same cavity setup already scripted) and check a
   large-`dt` run stays stable (the point of implicit momentum). Compare u(y) at x=0.5.

Files: `system.py` (predictor + PISO loop + a_P/H), a new momentum-matrix assembly
(triplets), reuse `fvm_utils_compute.py` face flux / gg_grad. Keep `scheme`/backend a
user choice. The current explicit-momentum path stays as the default/simple mode.

## Phase 2 (interFoam-style VOF, on top of this)
- ⬜ phase fraction `alpha` transport with bounded interface compression (MULES-like)
- ⬜ variable density/viscosity `rho(alpha), mu(alpha)`
- ⬜ surface tension (CSF: `sigma * kappa * grad(alpha)`, curvature `kappa`)
- ⬜ gravity/buoyancy; validate on the dam-break vs `interFoam`
