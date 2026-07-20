# Adding a new solver to manapy

A field guide for writing a new finite-volume solver (e.g. a new conservation
law) **without reinventing** what already exists. Distilled from building the
nonlinear viscous **Burgers** solver, which is the worked example throughout.

> TL;DR — a new explicit FV solver in manapy is usually **~15 lines of genuinely
> new code** (the numerical flux) plus assembly. Everything else — mesh, fields,
> reconstruction, limiter, diffusion, halos/MPI, time step, time loop — is
> reused. The convective kernel is **flux-function-agnostic**: it reconstructs
> the left/right face states and calls a swappable `_compute_flux`.

## The three load-bearing abstractions

Every solver stands on the same three "god-nodes" (see the graphify graph):

| Abstraction | File | Role | Answers |
|---|---|---|---|
| **`Domain`** | `domain/DomainClass.py` | assembles the partitioned + distributed mesh (cells/faces/nodes/halos + MPI comm + backend) | *on which geometry?* |
| **`compile()`** | `backends/compile_fun.py` | JIT of a kernel, backend-agnostic (python / numba CPU / GPU CUDA), MPI-synchronised cache | *with what code, on what hardware?* |
| **`Variable`** | `core/Variable.py` | a discretised field (`.cell/.face/.node/.ghost/.halo`) + FV operators (gradients, interpolations, halo/ghost exchange, BCs) | *what values, moved how?* |

A solver is glue between them: build `Variable`s on a `Domain`, and advance them
with `compile()`-d flux kernels.

## The recipe

### 0. Ask the graph first
`/graphify query "how does the <closest> solver compute its convective flux?"`
Find the nearest existing solver and the god-nodes before reading anything.

### 1. Classify the PDE
Hyperbolic / parabolic / elliptic? scalar or system? linear or nonlinear flux?
inviscid or viscous? 2D/3D? explicit or implicit?

### 2. Pick the template to clone
| Your PDE | Clone | Why |
|---|---|---|
| scalar advection–diffusion | `solvers/advecdiff` | convective + diffusive terms, MUSCL, BCs |
| pure hyperbolic scalar | `solvers/advec` | minimal explicit scalar FV |
| nonlinear system (compressible) | `solvers/euler` | Rusanov/Riemann flux + WENO, systems |
| steady linear (Laplace/Poisson/Darcy) | `solvers/ls` + `PoissonModel` | implicit linear solve |

### 3. Reuse (don't rewrite)
- `Domain`, `Variable` and its operators (`compute_cell_gradient`, `compute_face_gradient`,
  `interpolate_*`, `update_halo_value`, `update_ghost_value`, `_update_boundaries`).
- `compile()` / `compile_no_cache()` for kernels.
- The **flux-agnostic convective kernel** (`_explicitscheme_convective_2d/3d`) — it does
  the MUSCL reconstruction; you only supply the flux.
- The flux-agnostic diffusion / time-step / cell-update kernels
  (`_explicitscheme_dissipative`, `_time_step`, `_update_new_value`) — **import them**
  from `advecdiff.fvm_utils_compute`; they carry no flux global, so they are safe to reuse verbatim.

### 4. Write the only new thing: the numerical flux
Implement `_compute_flux(w_l, w_r, face_normal, flux_w)` for your physical flux
`f(u)`. `face_normal` is **area-weighted**, so the normal flux is `f(u)·(nx+ny[+nz])`.
Rusanov / local Lax-Friedrichs (robust, monotone for a convex scalar flux):

```
F = 0.5*(f_L + f_R) - 0.5*alpha*(w_r - w_l),   alpha = max|f'(u)·n|
```

Burgers example (`f(u)=u^2/2`): `alpha = |s|*max(|w_l|,|w_r|)` with `s = nx+ny+nz`.

> **Gotcha — the `_compute_flux` global is per-module.** The convective kernel
> inlines a module-level `_compute_flux` that `setup()` rebinds. Do **not** reuse
> advecdiff's compiled convective kernel with a different flux — you would corrupt
> advecdiff for other solvers in the same process. Keep a **local copy** of the
> convective kernel in your solver's `fvm_utils_compute.py` (this is why every
> solver already has its own copy). Compile the flux + convective kernel with
> `compile_no_cache` (they depend on the rebound global; a disk cache keyed on
> source alone could reuse a stale flux binding).

### 5. Files to create
```
solvers/<name>/__init__.py            # export the solver
solvers/<name>/system.py              # <Name>Solver: stepper / compute_fluxes / compute_new_val
solvers/<name>/fvm_utils_compute.py   # flux body + local convective kernel + setup(dim, scheme)
api/models.py                         # add a <Name>Model next to AdvectionModel (plugs into run())
examples/2D/<name>/<name>2d.py        # a runnable case
```
The solver must expose `stepper()`, `compute_fluxes()`, `compute_new_val()` so it
drops straight into `api/models._ExplicitModel.run(T, ...)`.

Self-advecting flux (Burgers): the solution *is* the transport speed, so pass
`var.cell` as the velocity to `_time_step` — the per-face CFL wave speed becomes
`|u·(nx+ny[+nz])|`.

### 6. Build the example through the high-level API (not low-level wiring)
```python
from manapy.api.mesh import Mesh
from manapy.api.models import BurgersModel
import numpy as np

mesh = Mesh.rectangle(bounds=((0,1),(0,1)), n=(200,40))   # on-the-fly, no .msh
u = mesh.field("u", init=lambda x,y,z: np.where(x<0.25, 1., 0.),
               bc={"in": ("dirichlet",1), "out": ("dirichlet",0),
                   "upper":"neumann", "bottom":"neumann"}, limiter="vanalbada")
BurgersModel(u, mesh, nu=0.01, order=2, cfl=0.4).run(T=1.0)
```
Patch names from the generators: `in`=x-min, `out`=x-max, `bottom`=y-min,
`upper`=y-max, `front`=z-min, `back`=z-max.

### 7. Verify against an EXACT solution (not just "it runs")
- **Shock speed** — Rankine–Hugoniot `s=(u_L+u_R)/2`; check the front position.
- **Full profile (viscous)** — the exact travelling wave
  `u = (u_L+u_R)/2 - (u_L-u_R)/2 · tanh[(u_L-u_R)(x-x0-st)/(4ν)]`; init with it, evolve,
  compare (`validate_burgers2d.py`).
- **Convergence** — refine and measure the L2 order.
  Expect **~1st order at a shock/viscous front** (the TVD limiter clips to 1st
  order there and all the error concentrates in the layer) even for a formally
  2nd-order MUSCL scheme; the order-2 reconstruction still ~halves the error.
  A clean ~2 shows up only on a fully smooth solution.
- **Monotonicity** — no overshoot outside `[u_R, u_L]`.
- **Plot** — `plot_burgers2d.py` overlays exact vs simulated + the pointwise error.

## Worked example: the Burgers solver
Files: `solvers/burgers/{__init__,system,fvm_utils_compute}.py`,
`api/models.BurgersModel`, `examples/2D/burgers/{burgers_riemann2d,validate_burgers2d,plot_burgers2d}.py`.
Only the ~15-line Rusanov flux was new; the MUSCL reconstruction, diffusion,
time step and update were reused from `advecdiff`.

## Known limits / next steps
- Burgers is **CPU-only, 2D** for now (3D kernel is a trivial copy of the 2D one;
  GPU needs a `cuda_fvm_utils.py` mirror as in `advecdiff`).
- Explicit + viscous ⇒ the **diffusive** stability limit `dt ~ h²/ν` dominates on
  fine meshes; go implicit for the diffusion term to relax it.
