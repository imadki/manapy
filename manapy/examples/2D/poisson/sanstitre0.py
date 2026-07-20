#!/usr/bin/env python3
"""darcy2d SOLVER-COMBO probe. Build the domain / variables / transport solver
ONCE, then sweep a list of pressure linear-solver configs. For each combo:
  * reset ne to the Gaussian IC AND reset P.cell to 0 (identical cold start),
  * build a FRESH pressure solver,
  * run PROBE_STEPS timed Darcy steps (solve P + advect),
  * record wall / per-step / L2,
  * DESTROY the solver (del + gc.collect + Barrier) so nothing leaks into the next.

Set MANAPY_GINKGO_PROFILE=1 to get the per-step `converged=/iters=` lines from
Ginkgo. Combos come from COMBOS below. Env: MESH_FILE, DT, PROBE_STEPS, DEVICE.
"""
import os, gc
import numpy as np
from mpi4py import MPI
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.solvers.ls import PETScKrylovSolver, MUMPSSolver, GinkgoDistributedSolver

COMM = MPI.COMM_WORLD; RANK = COMM.Get_rank()
PIN=float(os.environ.get("PIN",1.0)); SIGMA0=float(os.environ.get("SIGMA0",0.10))
X0=float(os.environ.get("X0",0.25)); Y0=float(os.environ.get("Y0",0.5))
CFL=float(os.environ.get("CFL",0.8)); ORDER=int(os.environ.get("ORDER",2))
SCHEME=os.environ.get("SCHEME","diamond"); MESH_FILE=os.environ["MESH_FILE"]
DT=float(os.environ.get("DT","2.828854e-04")); NSTEPS=int(os.environ.get("PROBE_STEPS","8"))
DEVICE=os.environ.get("DEVICE","omp").lower()
IMAX=int(os.environ.get("GK_IMAX",20000)); EPSR=float(os.environ.get("GK_EPSR",1e-10))
S2=SIGMA0**2; VX=PIN
def exact(xc,yc,t): return np.exp(-(((xc-X0-VX*t)**2+(yc-Y0)**2))/(2.0*S2))

AMG_ARGS = {
    "type": "solver::Bicgstab",
    "preconditioner": {
        "type": "solver::Multigrid", "max_levels": 10, "min_coarse_rows": 2,
        "mg_level": [{"type": "multigrid::Pgm", "deterministic": True}],
        "pre_smoother": [{"type": "solver::Ir", "relaxation_factor": 0.9,
            "solver": {"type": "preconditioner::Schwarz",
                        "local_solver": {"type": "preconditioner::Jacobi"}},
            "criteria": [{"type": "Iteration", "max_iters": 2}]}],
        "post_uses_pre": True,
        "coarsest_solver": {"type": "solver::Cg", "criteria": [{"type": "Iteration", "max_iters": 4}]},
        "default_initial_guess": "zero",
        "criteria": [{"type": "Iteration", "max_iters": 1}]},
    "criteria": [{"type": "Iteration", "max_iters": 1000},
                  {"type": "ResidualNorm", "reduction_factor": 1e-8}],
}

# name -> factory(domain, P)
COMBOS = [
    ("cg/none",   lambda d,P: GinkgoDistributedSolver(domain=d,var=P,device=DEVICE,scheme=SCHEME,reuse_mtx=True,verbose=False,method="cg",precond="none",  i_max=IMAX,eps_r=EPSR)),
    ("cg/jacobi", lambda d,P: GinkgoDistributedSolver(domain=d,var=P,device=DEVICE,scheme=SCHEME,reuse_mtx=True,verbose=False,method="cg",precond="jacobi",i_max=IMAX,eps_r=EPSR)),
    ("amg",       lambda d,P: GinkgoDistributedSolver(domain=d,var=P,device=DEVICE,scheme=SCHEME,reuse_mtx=True,verbose=False,solver_args=AMG_ARGS)),
    ("petsc/gamg",lambda d,P: PETScKrylovSolver(domain=d,var=P,scheme=SCHEME,reuse_mtx=True,method="gmres",precond="gamg",eps_a=1e-10,eps_r=1e-10)),
]

# ---- build domain + variables + transport ONCE ----
domain = Domain.create_domain(MESH_FILE, 2, Partitioning.Par_Nodal, recreate=True)
be = domain.backend; cells = domain.cells
boundaries = {"in":"dirichlet","out":"dirichlet","upper":"neumann","bottom":"neumann"}
values = {"in":PIN,"out":0.0}
ne=Variable(domain=domain,name="ne"); u=Variable(domain=domain,name="u"); v=Variable(domain=domain,name="v")
P =Variable(domain=domain,BC=boundaries,values_dict=values,name="P")
S = AdvectionDiffusionSolver(ne, vel=(u,v), Dxx=0.0, Dyy=0.0, order=ORDER, cfl=CFL)
ctr_h = np.asarray(be.to_host(cells.center)); vol_h = np.asarray(be.to_host(cells.volume))
ic_h  = np.ascontiguousarray(exact(ctr_h[:,0], ctr_h[:,1], 0.0), dtype=np.float64)
pcell_zero = np.zeros_like(np.asarray(be.to_host(P.cell)))

def darcy_step(L):
    L(); P.update_halo_value(); P.update_ghost_value(); P.interpolate_celltonode()
    L.compute_Sol_gradient()
    be.copy(u.face,P.gradfacex); be.copy(v.face,P.gradfacey)
    u.interpolate_facetocell(); v.interpolate_facetocell()
    S.compute_fluxes(); S.compute_new_val()

results=[]
for name, factory in COMBOS:
    be.copy(ne.cell, ic_h)              # reset scalar IC
    be.copy(P.cell, pcell_zero)         # reset pressure -> identical COLD start
    S.dt = DT
    if RANK==0: print(f"\n### COMBO {name} (steps={NSTEPS}, dt={DT:.3e}) ###", flush=True)
    L = None
    try:
        L = factory(domain, P)
        COMM.Barrier(); t0 = MPI.Wtime()
        for _ in range(NSTEPS):
            darcy_step(L)
        COMM.Barrier(); wall = COMM.reduce(MPI.Wtime()-t0, op=MPI.MAX, root=0)
        ne_h = np.asarray(be.to_host(ne.cell))
        err = ne_h - exact(ctr_h[:,0], ctr_h[:,1], NSTEPS*DT)
        num = COMM.reduce(float(np.sum(vol_h*err**2)), op=MPI.SUM, root=0)
        den = COMM.reduce(float(np.sum(vol_h)),         op=MPI.SUM, root=0)
        if RANK==0:
            results.append((name, wall, wall/NSTEPS, (num/den)**0.5))
            print(f"### DONE {name}: wall={wall:.3f}s per_step={wall/NSTEPS:.4f}s L2={ (num/den)**0.5:.3e}", flush=True)
    except Exception as e:
        if RANK==0:
            results.append((name, float('nan'), float('nan'), float('nan')))
            print(f"### FAILED {name}: {type(e).__name__}: {e}", flush=True)
    finally:
        del L; gc.collect(); COMM.Barrier()   # DESTROY before next combo

if RANK==0:
    print("\n==================== SOLVER COMBO SUMMARY ====================")
    print(f"{'combo':14s} {'wall(s)':>9s} {'per_step(s)':>12s} {'L2':>12s}")
    for n,w,ws,l2 in results:
        print(f"{n:14s} {w:9.3f} {ws:12.4f} {l2:12.3e}")
