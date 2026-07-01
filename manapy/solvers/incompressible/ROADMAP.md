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
- 🟡 **serial only**: the two-point Laplacian is assembled and factorised with SciPy
  (`splu`). Next: assemble the same operator through the distributed linear solvers
  (`solvers/ls/`, PETSc) so the projection runs under MPI like the rest of manapy.
- ⬜ this is a single-corrector Chorin projection, **not PISO**. icoFoam is PISO
  (momentum predictor + `nCorrectors` pressure loops with the a_P coefficients and
  Rhie-Chow). A PISO restructuring would tighten the pressure-velocity coupling.
- ⬜ higher-order convection (currently first-order upwind); non-orthogonal correction
  in the two-point face gradient (currently the orthogonal approximation).
- ⬜ 3D.

## Phase 2 (interFoam-style VOF, on top of this)
- ⬜ phase fraction `alpha` transport with bounded interface compression (MULES-like)
- ⬜ variable density/viscosity `rho(alpha), mu(alpha)`
- ⬜ surface tension (CSF: `sigma * kappa * grad(alpha)`, curvature `kappa`)
- ⬜ gravity/buoyancy; validate on the dam-break vs `interFoam`
