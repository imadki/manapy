#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darcy 2D benchmark (GPU): Darcy (potential) flow + passive scalar transport.

GPU twin of driver_manapy.py -- same Darcy setup run on the CUDA backend
(GPUBackend + the device kernels). Every step re-solves the steady Darcy pressure
laplacian(P)=0 (linear -> P = PIN*(1-x)), rebuilds the Darcy velocity U = grad(P) =
(PIN, 0), then advects a Gaussian pulse one step. Since U is constant the exact
solution is a pure translation of the initial Gaussian.

Fast flux path: only U.n at the faces is needed (the compact two-point pressure
difference gives it exactly), so we skip the cell-gradient reconstruction and the
node/face interpolation -- the OpenFOAM pEqn.flux() equivalent, L2 bit-identical.

Pressure solver on the device: Ginkgo CG, NO preconditioner (warm-started x = steady
pressure converges in ~2 iters; a block-Jacobi setup was 82% of the GPU time). 1st-order
upwind advection (order=1), diamond pressure-gradient scheme (triangle mesh).

Initial / exact field:
    ne(x,y,t) = exp(-((x - X0 - PIN*t)^2 + (y - Y0)^2) / (2*sigma0^2))
    U = grad(P) = (PIN, 0),  P = PIN*(1 - x)

Run:
    MESH_DIR=../../../../meshes/geo MESH_FILE=square.msh mpirun -n 1 python3 darcy2d_gpu.py
"""
from mpi4py import MPI
import os
import timeit
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.solvers.ls import GinkgoDistributedSolver
from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = os.environ.get('MESH_FILE', 'square.msh')
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

# --- GPU backend ---
gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=False)
gpu.init_stream()
gpu.set_config(free=True, nb_threads=128)   # 256 threads/bloc au lieu de 32

backend = gpu

start = timeit.default_timer()
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True,
                              backend=backend)
end = timeit.default_timer()
tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

cells = domain.cells
nbcells = domain.nbcells

# physical parameters
PIN = float(os.environ.get('PIN', '1.0'))     # inlet pressure -> constant velocity (PIN,0)
sigma0 = float(os.environ.get('SIGMA0', '0.10'))
x0, y0 = float(os.environ.get('X0', '0.25')), float(os.environ.get('Y0', '0.5'))
cfl = float(os.environ.get('CFL', '0.8'))
tfinal = float(os.environ.get('TFINAL', '0.25'))
s2 = sigma0 ** 2
vx = PIN                                        # U = grad(P) = (PIN, 0)

# DARCY_FLUX_MODE=normal (default here): fv-only fast path -- only U.n at the faces,
# skips the vector velocity reconstruction + node/face interpolation (~2x cheaper,
# L2 bit-identical). =vector: full reconstruction (the CPU-driver default).
flux_mode = os.environ.get('DARCY_FLUX_MODE', 'normal').lower()

# Fixed time step (set DT=... to match a controller=none run); DT unset / 0 falls back
# to the hard-coded default below.
fixed_dt = float(os.environ.get('DT', '0') or 0.0)

# transport scalar + Darcy velocity + pressure (Dirichlet in/out, Neumann walls)
boundaries = {"in": "dirichlet", "out": "dirichlet",
              "upper": "neumann", "bottom": "neumann"}
values = {"in": PIN, "out": 0.0}
ne = Variable(domain=domain, name="ne")
u = Variable(domain=domain, name="u")
v = Variable(domain=domain, name="v")
P = Variable(domain=domain, BC=boundaries, values_dict=values, name="P")

# 1st-order upwind advection, diamond pressure-gradient scheme (triangle mesh).
S = AdvectionDiffusionSolver(ne, vel=(u, v), Dxx=0.0, Dyy=0.0, order=1, cfl=cfl)
# Ginkgo CG on CUDA, no preconditioner (fresh-solver path, warm-started ~2 iters).
L = GinkgoDistributedSolver(domain=domain, var=P, device="cuda", scheme="diamond",
                            reuse_mtx=True, verbose=False, spd=False,
                            method="cg", precond="none",
                            i_max=int(os.environ.get('GK_IMAX', '20000')),
                            eps_r=float(os.environ.get('GK_EPSR', '1e-10')))


def exact_ne(centers, t):
  xc = centers[:, 0]
  yc = centers[:, 1]
  return np.exp(-(((xc - x0 - vx * t) ** 2 + (yc - y0) ** 2)) / (2.0 * s2))


# --- initialisation (Gaussian pulse) --- computed on host, pushed to the device Variable.
centers_h = np.asarray(gpu.to_host(cells.center))
ne_h = exact_ne(centers_h, 0.0)
gpu.copy(ne.cell, gpu.to_device(np.ascontiguousarray(ne_h, dtype=np.float64)))

COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Computation ...")

time = 0.0
niter = 0

d_t = fixed_dt if fixed_dt > 0.0 else 2.000000e-04
while time < tfinal:
  S.dt = d_t
  time += d_t

  # solve steady Darcy pressure, then rebuild the advecting velocity
  L()
  P.update_halo_value()
  P.update_ghost_value()
  if flux_mode == "normal":                    # fast: only U.n at the faces
    L.compute_Sol_gradient(normal_only=True)
    gpu.copy(u.face, P.gradfacex)
    gpu.copy(v.face, P.gradfacey)
  else:                                        # vector: full velocity reconstruction
    P.interpolate_celltonode()
    L.compute_Sol_gradient()
    gpu.copy(u.face, P.gradfacex)
    gpu.copy(v.face, P.gradfacey)
    u.interpolate_facetocell()
    v.interpolate_facetocell()

  S.compute_fluxes()
  S.compute_new_val()
  niter += 1

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# --- L2 error vs analytic (volume-weighted, OWNED cells only) --- copied to host.
ne_hs = np.asarray(gpu.to_host(ne.cell))[:nbcells]
vol = np.asarray(gpu.to_host(cells.volume))[:nbcells]
cen = np.asarray(gpu.to_host(cells.center))[:nbcells]
err = ne_hs - exact_ne(cen, time)
local_num = float(np.sum(vol * err * err))
local_den = float(np.sum(vol))
num = COMM.allreduce(local_num, op=MPI.SUM)
den = COMM.allreduce(local_den, op=MPI.SUM)
glob_cells = COMM.allreduce(nbcells, op=MPI.SUM)
l2 = np.sqrt(num / den)

if RANK == 0:
  mode = "fixed" if fixed_dt > 0.0 else "default"
  print(f"mesh={filename} cells={glob_cells} iters={niter} dt={d_t:.6e} ({mode}) "
        f"flux={flux_mode} tfinal={time:.6f} walltime={walltime:.4f}s L2(ne)={l2:.6e}")
