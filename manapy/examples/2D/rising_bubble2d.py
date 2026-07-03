#!/usr/bin/env python3
"""Rising bubble benchmark, Hysing et al. (2009) test case 1, on triangles.
Domain [0,1]x[0,2]; bubble r=0.25 at (0.5,0.5): rho 1000/100, mu 10/1, g=0.98,
sigma=24.5 (Re=35, Eo=10). alpha=1 heavy fluid, alpha=0 bubble; exact cut-cell
fractions via tangent half-planes of the circle.
References (TP2D/FreeLIFE/MooNMD): max rise velocity ~0.2417 at t~0.9-1.0,
centroid y_c(t=3) ~ 1.081.
Measured (h~1/50 triangles, dt=2e-3, 4 ranks): v_max = 0.205 at t=1.04,
y_c(3) = 1.012, phase volume conserved to 1e-6. The under-prediction is expected:
the benchmark prescribes FREE-SLIP lateral walls (not supported yet -> no-slip here,
extra drag) and the momentum convection is first-order upwind.

Run:
    mpirun -np 4 python3 rising_bubble2d.py
"""
from mpi4py import MPI
import os, numpy as np
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.system import IncompressibleSolver
from manapy.solvers.incompressible.vof import volume_fractions

RANK = MPI.COMM_WORLD.Get_rank(); COMM = MPI.COMM_WORLD
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'meshes', 'geo')
mesh = os.environ.get('MESH', os.path.join(BASE, 'rect12.msh'))

xc0, yc0, R = 0.5, 0.5, 0.25
dom = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
u = Variable(domain=dom); v = Variable(domain=dom)
P = Variable(domain=dom,
             BC={"in": "neumann", "out": "neumann", "bottom": "neumann", "upper": "dirichlet"},
             values_dict={"upper": 0.0})
alpha = Variable(domain=dom)
# bubble = intersection of N tangent half-planes of the circle (convex): fraction INSIDE
N = 96
planes = [(np.cos(th), np.sin(th), np.cos(th) * xc0 + np.sin(th) * yc0 + R)
          for th in np.linspace(0, 2 * np.pi, N, endpoint=False)]
alpha.cell[:] = 1.0 - volume_fractions(dom, planes)      # alpha=1 heavy, 0 in bubble
zbc = {"in": 0.0, "out": 0.0, "upper": 0.0, "bottom": 0.0}


solver = IncompressibleSolver(u, v, P, ncorr=3, n_outer=int(os.environ.get('NOUTER', 2)),
                              implicit_momentum=True, u_bc=zbc, v_bc=zbc, nu=1e-6,
                              conv_order=int(os.environ.get('CONV', 2)),
                              alpha=alpha, rho1=1000.0, rho2=100.0, mu1=10.0, mu2=1.0,
                              gravity=(0.0, -0.98), sigma=24.5, cAlpha=1.0)

dt = float(os.environ.get('DT', 2e-3)); tend = float(os.environ.get('TEND', 3.0))
nsteps = int(round(tend / dt))
nvtk = int(os.environ.get('NVTK', 10))                  # VTK snapshots over the run
vtk_every = max(1, nsteps // nvtk)

# like interFoam: just step and write the fields. Benchmark quantities (y_c, v_rise)
# are post-processed from the VTK afterwards (see bubble_bench/postproc.py).
t = 0.0; k = 0; nsave = 0
while t < tend - 1e-12:
    solver.step(dt=dt)
    t += dt; k += 1
    if k % vtk_every == 0 and nsave < nvtk:
        dom.save_on_cell_multi(["u", "v", "P", "alpha"],
                               [u.cell, v.cell, P.cell, alpha.cell], dt, t, k, nsave)
        nsave += 1
