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

## Phase 2 (interFoam-style VOF, on top of this)
- ⬜ phase fraction `alpha` transport with bounded interface compression (MULES-like)
- ⬜ variable density/viscosity `rho(alpha), mu(alpha)`
- ⬜ surface tension (CSF: `sigma * kappa * grad(alpha)`, curvature `kappa`)
- ⬜ gravity/buoyancy; validate on the dam-break vs `interFoam`
