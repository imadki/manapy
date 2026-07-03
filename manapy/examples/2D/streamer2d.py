#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D streamer discharge (plasma) solver, modern manapy API.

Couples, every time step:
  - a Poisson problem for the electric potential P (Ginkgo distributed solver),
    with source term = charge density 1.8096e-8 * (ne - ni);
  - drift-diffusion transport of the electron density ne (convective + diffusive
    + ionization source);
  - the ion density ni evolves by the ionization source only (ions immobile).
The electric field E = grad(P) drives the electron drift velocity (mobility
tables in solvers/streamer/fvm_utils_compute.py).

CPU reference run (default backend). The GPU port mirrors advecdiff/darcy2d.
"""
from mpi4py import MPI
import timeit
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.solvers.streamer.tools_utils_compute import initialisation_streamer_2d
from manapy.solvers.streamer.system import StreamerSolver
from manapy.solvers.ls import GinkgoDistributedSolver, MUMPSSolver, PETScKrylovSolver
from manapy.core.Variable import Variable
from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'mesh')

filename = os.environ.get('STREAMER_MESH', 'rectangle_st.msh')

_NITER_MAX = int(os.environ.get('STREAMER_NITER_MAX', '0'))  # 0 = no cap (debug)

dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=True)
gpu.init_stream()
gpu.set_config(free=True)

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, #recreate=True,
                              # backend=gpu
                              )
be = domain.backend
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

end = timeit.default_timer()
tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

if RANK == 0: print("Start Computation ...")
time = 0
tfinal = 5.25e-8
miter = 0
niter = 1
Pinit = 25000.
saving_at_node = 1
order = 2
cfl = 0.4

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "neumann",
              "bottom": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
ni = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
Ex = Variable(domain=domain)
Ey = Variable(domain=domain)
P = Variable(domain=domain, BC=boundaries, values_dict=values)

# Streamer transport + coupling solver (ne/ni, velocity (u,v), field (Ex,Ey), P).
S = StreamerSolver(ne, ni, vel=(u, v), E=(Ex, Ey), P=P, De=0., order=order, cfl=cfl)

# Initialisation on host (njit), then copy into the backend fields (device on GPU).
_ne = np.zeros(nbcells); _ni = np.zeros(nbcells)
_u = np.zeros(nbcells); _v = np.zeros(nbcells)
_Ex = np.zeros(nbcells); _Ey = np.zeros(nbcells); _P = np.zeros(nbcells)
initialisation_streamer_2d(_ne, _ni, _u, _v, _Ex, _Ey, _P, be.to_host(cells.center), Pinit)
be.copy(ne.cell, _ne); be.copy(ni.cell, _ni)
be.copy(u.cell, _u); be.copy(v.cell, _v)
be.copy(Ex.cell, _Ex); be.copy(Ey.cell, _Ey); be.copy(P.cell, _P)

L = MUMPSSolver(domain=domain, var=P, reuse_mtx=True, 
                scheme='fv_corrected', 
                non_orthogonal_corrections=0)

# L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=True, scheme='fv_corrected',
#               precond='gamg', sub_precond="amg",  # with_mtx=False,
#               eps_a=1e-10, eps_r=1e-10, method="gmres")

# # Ginkgo bicgstab (the diamond matrix is non-symmetric, so NOT cg) preconditioned
# # by AMG: Multigrid/Pgm with a Schwarz(Jacobi) smoother and a Cg coarse solver.
# # AMG cuts the Krylov iteration count drastically vs unpreconditioned gmres/bicgstab.
# amg_args = {
#     "type": "solver::Bicgstab",
#     "preconditioner": {
#         "type": "solver::Multigrid", "max_levels": 10, "min_coarse_rows": 2,
#         "mg_level": [{"type": "multigrid::Pgm", "deterministic": True}],
#         "pre_smoother": [{"type": "solver::Ir", "relaxation_factor": 0.9,
#             "solver": {"type": "preconditioner::Schwarz",
#                         "local_solver": {"type": "preconditioner::Jacobi"}},
#             "criteria": [{"type": "Iteration", "max_iters": 2}]}],
#         "post_uses_pre": True,
#         "coarsest_solver": {"type": "solver::Cg",
#                             "criteria": [{"type": "Iteration", "max_iters": 4}]},
#         "default_initial_guess": "zero",
#         "criteria": [{"type": "Iteration", "max_iters": 1}]},
#     "criteria": [{"type": "Iteration", "max_iters": 1000},
#                   {"type": "ResidualNorm", "reduction_factor": 1e-8}],
# }
# L = GinkgoDistributedSolver(domain=domain, var=P, device="cuda", scheme='fv_corrected',
#                             reuse_mtx=True, verbose=False, solver_args=amg_args)

ts = MPI.Wtime()
if RANK == 0: print("Start While loop ...")

while time < tfinal:

  # Poisson: solve for P with the charge-density source (local owned rows).
  rhs = S.update_rhs()
  # ts = MPI.Wtime()
  L(rhs=rhs[:L.localsize])
  te = MPI.Wtime()
  
  # print('time solveur', te - ts)
  
  # if niter == 10:
  #     import sys; sys.exit()
  
  P.update_halo_value()
  P.update_ghost_value()
  P.interpolate_celltonode()
  L.compute_Sol_gradient()

  # Electric field from grad(P), then the electron drift velocity.
  S.compute_Electric_Field()
  S.compute_Velocity()

  # Explicit time step (convective + diffusive CFL).
  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1
  time = time + d_t

  # ne/ni transport: ionization source + convective + diffusive fluxes, update.
  S.compute_fluxes()
  S.compute_new_val()

  if _NITER_MAX and RANK == 0:
    hne = be.to_host(ne.cell); hP = be.to_host(P.cell)
    hEx = be.to_host(Ex.cell); hu = be.to_host(u.cell)
    print(f"iter={niter} time={time:.4e} dt={d_t:.4e} "
          f"max(ne)={hne.max():.4e} min(ne)={hne.min():.4e} "
          f"max(P)={hP.max():.4e} max(|Ex|)={np.abs(hEx).max():.4e} "
          f"max(|u|)={np.abs(hu).max():.4e}", flush=True)

  if niter == 1 or niter % tot == 0:
    if RANK == 0:
      print(f"iter={niter} time={time:.4e} dt={d_t:.4e} "
            f"max(ne)={be.to_host(ne.cell).max():.4e} max(P)={be.to_host(P.cell).max():.4e} "
            f"max(|Ex|)={np.abs(be.to_host(Ex.cell)).max():.4e}", flush=True)
    if saving_at_node:
      ne.update_halo_value(); ne.update_ghost_value(); ne.interpolate_celltonode()
      ni.update_halo_value(); ni.update_ghost_value(); ni.interpolate_celltonode()
      u.update_halo_value(); u.update_ghost_value(); u.interpolate_celltonode()
      domain.save_on_node_multi(["ne", "ni", "u", "P"],
                                [be.to_host(ne.node), be.to_host(ni.node),
                                 be.to_host(u.node), be.to_host(P.node)],
                                d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "ni", "u", "P"],
                                [be.to_host(ne.cell), be.to_host(ni.cell),
                                 be.to_host(u.cell), be.to_host(P.cell)],
                                d_t, time, niter, miter)
    miter += 1

  niter += 1
  if _NITER_MAX and niter > _NITER_MAX:
    break

te = MPI.Wtime()
tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)
