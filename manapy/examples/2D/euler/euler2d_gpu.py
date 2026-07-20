#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manapy side of the Euler 2D benchmark — entropy wave (Gaussian density pulse).

2D twin of the euler3d driver. With uniform pressure and velocity the density
perturbation is simply advected at the flow speed without distortion, so the exact
solution is a translated Gaussian:
    rho(x,y,t) = 1 + 5*exp(-((x-x0-a*t)^2 + (y-y0)^2)/sigma^2)
    u = a,  v = 0,  p = 1,  gamma = 1.4
Mean flow is supersonic and the pulse stays interior, so Neumann ghost boundaries
never matter. The IC is identical to the one pyBaram and OpenFOAM are initialised
with (same a, sigma, x0/y0), so the three codes differ ONLY in spatial reconstruction.

Two SEPARATE entry paths (aligned with the euler3d driver — no per-iteration
branching mixing them):

    MODE=dt   dt provider   : build once, warm the numba (disk) cache with a few
                              steps, print "manapy dt=<value>" and stop. Fast — it
                              does NOT integrate to tfinal. Used by case.sh dt_for;
                              dt = DT-override if DT>0 else the CFL(0.5) step.
    MODE=run  comparison    : timed run at the shared FIXED dt, then volume-weighted
                              L2(rho) vs the exact solution. Default mode.

Env knobs (mirror the other benchmark drivers):
    MESH_FILE  shared 2D triangle .msh (required; e.g. MESH_FILE=uns_square.msh; same file fed to pyBaram / OpenFOAM)
    ORDER      1 (upwind) or 2 (MUSCL)                                  (default 1)
    DT         fixed dt. MODE=run requires DT>0 (from the dt provider).
    TFINAL     final time                                              (default 0.25)
    MODE       dt | run   (default run)
"""
import os
import numpy as np
from mpi4py import MPI

from manapy.domain import Domain, Partitioning
from manapy.solvers.euler.system import EulerSolver
from manapy.core.Variable import Variable

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# physical parameters (shared by every code — must match pyBaram / OpenFOAM ICs)
GAMMA = 1.4
A = 2.0                       # advection speed (x)
SIGMA = 0.05
X0, Y0 = 0.2, 0.2
P0 = 1.0
ORDER = int(os.environ.get("ORDER", "1"))
WARMUP_STEPS = 3              # enough to JIT-compile every kernel into the disk cache


def exact_rho(cells, t):
    xc, yc = cells.center[:, 0], cells.center[:, 1]
    return 1.0 + 5.0 * np.exp(-((xc - X0 - A * t) ** 2 + (yc - Y0) ** 2) / SIGMA ** 2)


def build():
    """Domain + Euler solver + entropy-wave initial condition (shared by both modes)."""
    domain = Domain.create_domain(os.environ["MESH_FILE"], 2,
                                  Partitioning.Par_Nodal, recreate=True)
    cells = domain.cells

    rho = Variable(domain=domain)
    P = Variable(domain=domain)
    rhou = Variable(domain=domain)
    rhov = Variable(domain=domain)
    rhoE = Variable(domain=domain)

    S = EulerSolver(rho, P, rhou, rhov, rhoE, gamma=GAMMA, cfl=0.5,
                    order=ORDER, scheme="rusanov", bc="Neumann")

    rho.cell[:] = exact_rho(cells, 0.0)
    P.cell[:] = P0
    rhou.cell[:] = rho.cell[:] * A
    rhov.cell[:] = 0.0
    rhoE.cell[:] = 0.5 * rho.cell[:] * A ** 2 + P.cell[:] / (GAMMA - 1.0)
    return domain, S, rho


def provide_dt():
    """dt PROVIDER: report the time step and warm the numba cache. No full run."""
    _, S, _ = build()
    fixed_dt = float(os.environ.get("DT", "0") or 0.0)
    dt = fixed_dt if fixed_dt > 0.0 else S.stepper()   # stepper() = CFL(0.5) dt (also JITs it)
    S.dt = dt
    for _ in range(WARMUP_STEPS):                       # JIT the flux/update kernels
        S.compute_fluxes(t=0.0)
        S.compute_new_val()
    if RANK == 0:
        print(f"manapy dt={dt:.6e}")


def run_benchmark():
    """COMPARISON run: timed loop at a fixed dt, then L2(rho) vs exact."""
    domain, S, rho = build()
    cells = domain.cells
    tfinal = float(os.environ.get("TFINAL", 0.25))
    dt = float(os.environ.get("DT", "0") or 0.0)
    if dt <= 0.0:
        raise SystemExit("MODE=run needs a fixed DT>0 (obtain it via MODE=dt)")
    S.dt = dt

    COMM.Barrier()
    ts = MPI.Wtime()
    time = 0.0
    niter = 0
    step = dt
    while time < tfinal:
        if time + step > tfinal:            # only branch: trim the final step onto tfinal
            step = tfinal - time
            S.dt = step
        time += step
        S.compute_fluxes(t=time)
        S.compute_new_val()
        niter += 1
    walltime = COMM.reduce(MPI.Wtime() - ts, op=MPI.MAX, root=0)

    err = rho.cell[:] - exact_rho(cells, time)
    vol = cells.volume[:]
    num = COMM.allreduce(float(np.sum(vol * err * err)), op=MPI.SUM)
    den = COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
    glob_cells = COMM.allreduce(domain.nbcells, op=MPI.SUM)
    if RANK == 0:
        l2 = np.sqrt(num / den)
        print(f"manapy cells={glob_cells} order={ORDER} iters={niter} dt={dt:.6e} "
              f"tfinal={time:.6f} L2={l2:.6e} wall={walltime:.4f}s")


if __name__ == "__main__":
    if os.environ.get("MODE", "run") == "dt":
        provide_dt()
    else:
        run_benchmark()
