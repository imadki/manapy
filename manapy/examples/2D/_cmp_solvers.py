#!/usr/bin/env python3
# Compare MUMPS vs Ginkgo on identical Poisson, giving EACH solver a FRESH copy
# of the source (MUMPS mutates its rhs argument in place!).
from mpi4py import MPI
import os
import numpy as np
from manapy.domain import Domain, Partitioning
from manapy.solvers.streamer.tools_utils_compute import initialisation_streamer_2d
from manapy.solvers.ls import GinkgoDistributedSolver, MUMPSSolver
from manapy.core.Variable import Variable

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
mesh = os.path.join(BASE, 'mesh', os.environ.get('STREAMER_MESH', 'rectangle_64K.msh'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
Pinit = 25000.
bc = {"in": "dirichlet", "out": "dirichlet", "upper": "neumann", "bottom": "neumann"}
vals = {"in": Pinit, "out": 0.}

ne = np.zeros(domain.nbcells); ni = np.zeros(domain.nbcells)
u = np.zeros(domain.nbcells); v = np.zeros(domain.nbcells)
Ex = np.zeros(domain.nbcells); Ey = np.zeros(domain.nbcells); Pc = np.zeros(domain.nbcells)
initialisation_streamer_2d(ne, ni, u, v, Ex, Ey, Pc, cells.center, Pinit)
ni = ni * 0.5
src = 1.8096e-8 * (ne - ni)


def solve(SolverCls, **kw):
    P = Variable(domain=domain, BC=bc, values_dict=vals)
    L = SolverCls(domain=domain, var=P, **kw)
    L(rhs=src[:L.localsize].copy())   # FRESH copy: MUMPS mutates its rhs in place
    return np.asarray(P.cell).copy()


Pm = solve(MUMPSSolver, reuse_mtx=True)
if RANK == 0:
    print(f"MUMPS            : max={Pm.max():.4e} min={Pm.min():.4e} mean={Pm.mean():.4e}")
for meth in ("cg", "bicgstab", "gmres"):
    Pg = solve(GinkgoDistributedSolver, device="omp", scheme="diamond", method=meth,
               reuse_mtx=True, verbose=False, eps_r=1e-9, i_max=5000)
    if RANK == 0:
        n = min(len(Pm), len(Pg)); a, b = Pm[:n], Pg[:n]
        rel = np.abs(a-b).max()/max(np.abs(a).max(), 1e-30)
        print(f"Ginkgo/{meth:9s}: max={b.max():.4e} min={b.min():.4e} "
              f"mean={b.mean():.4e}  rel.diff vs MUMPS = {rel:.3e}")
