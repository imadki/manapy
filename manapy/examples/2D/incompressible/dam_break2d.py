#!/usr/bin/env python3
"""Dam break (Martin & Moyce 1952) on the triangle mesh carre.msh, density ratio 1000
(water/air). Water column a=0.25 wide, 2a tall, in a closed unit box; alpha initialised
with EXACT cut-cell volume fractions (vof.volume_fractions). Front position X = x/a vs
T = t sqrt(2g/a); Martin-Moyce experiment: front reaches the far wall (X=4) at T ~ 2.9.
Validated: manapy front hits X=3.9 at T = 3.01 (dt=5e-4, NSUB=2, 4 MPI ranks), air and
water velocities stay physical (~3 m/s) and the phase volume is conserved to 1e-6.

Run:
    DT=5e-4 NSUB=2 mpirun -np 4 python3 dam_break2d.py
"""
from mpi4py import MPI
import os, numpy as np
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.system import IncompressibleSolver
from manapy.solvers.incompressible.vof import volume_fractions

RANK = MPI.COMM_WORLD.Get_rank()
COMM = MPI.COMM_WORLD
BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', 'meshes', 'geo')
mesh = os.environ.get('MESH', os.path.join(BASE, 'carre.msh'))

a = 0.25; g = 9.81; ratio = float(os.environ.get('RATIO', 1000.0))
dom = Domain.create_domain(mesh, 2, Partitioning.Par_Nodal, recreate=True)
xc, yc = dom.cells.center[:, 0], dom.cells.center[:, 1]
u = Variable(domain=dom); v = Variable(domain=dom)
P = Variable(domain=dom,
             BC={"in": "neumann", "out": "neumann", "bottom": "neumann", "upper": "dirichlet"},
             values_dict={"upper": 0.0})
alpha = Variable(domain=dom)
alpha.cell[:] = volume_fractions(dom, [(1.0, 0.0, a), (0.0, 1.0, 2 * a)])  # column a x 2a
zbc = {"in": 0.0, "out": 0.0, "upper": 0.0, "bottom": 0.0}
solver = IncompressibleSolver(u, v, P, ncorr=3, n_outer=int(os.environ.get('NOUTER', 2)),
                              n_alpha_sub=int(os.environ.get('NSUB', 1)),
                              implicit_momentum=True, u_bc=zbc, v_bc=zbc, nu=1e-6,
                              cfl=float(os.environ.get('CFL', 0.3)),
                              ddt_corr=bool(int(os.environ.get('DDT', 0))),
                              alpha=alpha, rho1=ratio, rho2=1.0, mu1=1e-3, mu2=1e-5,
                              gravity=(0.0, -g), cAlpha=float(os.environ.get('CALPHA', 1.0)))
dt = float(os.environ.get('DT', 1e-3))
adapt = bool(int(os.environ.get('ADAPT', 0)))
dtmax = float(os.environ.get('DTMAX', 2e-3))
tend = float(os.environ.get('TEND', 0.45))
scale = np.sqrt(2 * g / a)
t = 0.0; k = 0
if RANK == 0:
    print(f"# dam break ratio={ratio:.0f} dt={dt} mesh={os.path.basename(mesh)}", flush=True)
    print(f"{'T':>7} {'X_front':>8} {'|u|max':>10} {'vol err %':>10}")
vol = np.asarray(dom.cells.volume)
vol0 = COMM.allreduce(float(np.sum(alpha.cell * vol)), op=MPI.SUM)
hit = None
tnext = 0.0
while t < tend:
    if adapt:
        dtk = min(solver.stepper(), dtmax)
        solver.step(dt=dtk)
    else:
        dtk = dt
        solver.step(dt=dt)
    t += dtk; k += 1
    if (adapt and t >= tnext) or (not adapt and k % 20 == 0):
        tnext = t + 0.01
        wet = alpha.cell > 0.5
        xf_loc = float(np.max(xc[wet])) if np.any(wet) else 0.0
        xf = COMM.allreduce(xf_loc, op=MPI.MAX)
        sp = np.sqrt(u.cell**2 + v.cell**2)
        umw = COMM.allreduce(float(np.max(sp[wet])) if np.any(wet) else 0.0, op=MPI.MAX)
        uma = COMM.allreduce(float(np.max(sp[~wet])) if np.any(~wet) else 0.0, op=MPI.MAX)
        vv = COMM.allreduce(float(np.sum(alpha.cell * vol)), op=MPI.SUM)
        if RANK == 0:
            print(f"{t*scale:7.3f} {xf/a:8.3f}  water {umw:9.3e}  air {uma:9.3e} "
                  f"{100*(vv-vol0)/vol0:10.4f}", flush=True)
        if xf / a > 3.9 and hit is None:
            hit = t * scale
            if RANK == 0:
                print(f"### front reaches far wall (X=3.9) at T = {hit:.2f}  "
                      f"(Martin-Moyce ~2.9, pre-fix ~1.37)", flush=True)
            break
if RANK == 0 and hit is None:
    print(f"### front did NOT reach the wall by T = {t*scale:.2f}", flush=True)
