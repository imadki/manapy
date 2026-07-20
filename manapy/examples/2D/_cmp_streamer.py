#!/usr/bin/env python3
# Run the coupled streamer for a few iters with a chosen Poisson solver, to check
# whether Ginkgo (with an accurate solve) tracks the stable MUMPS behaviour.
from mpi4py import MPI
import os
import numpy as np
from manapy.domain import Domain, Partitioning
from manapy.solvers.streamer.tools_utils_compute import initialisation_streamer_2d
from manapy.solvers.streamer.system import StreamerSolver
from manapy.solvers.ls import GinkgoDistributedSolver, MUMPSSolver
from manapy.core.Variable import Variable

RANK = MPI.COMM_WORLD.Get_rank()
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
mesh = os.path.join(BASE, 'mesh', os.environ.get('STREAMER_MESH', 'rectangle_1K.msh'))
which = os.environ.get('SOLVER', 'ginkgo')
NMAX = int(os.environ.get('NMAX', '40'))
domain = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
Pinit = 25000.
bc = {"in": "dirichlet", "out": "dirichlet", "upper": "neumann", "bottom": "neumann"}
vals = {"in": Pinit, "out": 0.}

ne = Variable(domain=domain); ni = Variable(domain=domain)
u = Variable(domain=domain); v = Variable(domain=domain)
Ex = Variable(domain=domain); Ey = Variable(domain=domain)
P = Variable(domain=domain, BC=bc, values_dict=vals)
S = StreamerSolver(ne, ni, vel=(u, v), E=(Ex, Ey), P=P, De=0., order=2, cfl=0.4)
initialisation_streamer_2d(ne.cell, ni.cell, u.cell, v.cell, Ex.cell, Ey.cell,
                           P.cell, cells.center, Pinit)

if which == 'mumps':
    L = MUMPSSolver(domain=domain, var=P, reuse_mtx=True)
else:
    L = GinkgoDistributedSolver(domain=domain, var=P, device="omp", scheme="diamond",
                                method="gmres", reuse_mtx=True, verbose=False,
                                eps_r=1e-10, i_max=5000)

time = 0.0
for niter in range(1, NMAX + 1):
    rhs = S.update_rhs()
    L(rhs=rhs[:L.localsize])
    P.update_halo_value(); P.update_ghost_value(); P.interpolate_celltonode()
    L.compute_Sol_gradient()
    S.compute_Electric_Field(); S.compute_Velocity()
    d_t = S.stepper(); time += d_t
    S.compute_fluxes(); S.compute_new_val()
    if RANK == 0:
        print(f"[{which}] it={niter:3d} t={time:.4e} dt={d_t:.4e} "
              f"max(ne)={ne.cell.max():.4e} min(ne)={ne.cell.min():.4e} "
              f"max|Ex|={np.abs(Ex.cell).max():.4e}", flush=True)
