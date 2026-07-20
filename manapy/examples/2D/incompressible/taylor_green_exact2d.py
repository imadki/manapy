#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-solution test of the incompressible projection solver: Taylor-Green vortex.

The 2D Taylor-Green vortex is a closed-form unsteady solution of the incompressible
Navier-Stokes equations on [0, 2*pi]^2:

    u(x,y,t) =  sin(x) cos(y) exp(-2 nu t)
    v(x,y,t) = -cos(x) sin(y) exp(-2 nu t)
    p(x,y,t) = -(rho/4) (cos 2x + cos 2y) exp(-4 nu t)

div u = 0 exactly, and the kinetic energy decays as exp(-4 nu t). It exercises the
full solver -- convection, viscous diffusion AND the pressure projection.

manapy's IncompressibleSolver imposes a Dirichlet wall velocity per boundary
(no periodic BC), so we feed the EXACT velocity on every boundary face each step
(a manufactured-boundary exact test): the exact field then IS the solution of the
initial-boundary-value problem, and the interior must converge to it.

Run:
    python3 taylor_green_exact2d.py
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.core.Variable import Variable
from manapy.solvers.incompressible.system import IncompressibleSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---- parameters ----
Lx = 2.0 * np.pi
nu = 0.05
rho = 1.0
T = 2.0
nx = ny = 64


def exact_u(x, y, t):
  return np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)


def exact_v(x, y, t):
  return -np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)


# ---- mesh (api) + fields ----
# TRIANGLES work here thanks to the deferred non-orthogonal correction (n_nonorth>0):
# the orthogonal-only 'fv' operator alone stagnates at ~3% on triangles (order ~0.1),
# but correcting BOTH the pressure Laplacian AND the momentum viscous diffusion restores
# convergence (order ~0.9, same as quad). Set cell_type="quad" + n_nonorth=0 to compare.
mesh = Mesh.rectangle(bounds=((0.0, Lx), (0.0, Lx)), n=(nx, ny), 
                      cell_type="triangle")
domain = mesh.domain
cells, faces = domain.cells, domain.faces

u = mesh.field("u")
v = mesh.field("v")
# pressure: Neumann on 3 walls + one Dirichlet reference to fix the constant.
P = mesh.field("P", bc={"in": "neumann", "out": "neumann", "upper": "neumann",
                        "bottom": ("dirichlet", 0.0)})

# cfl=0.2: the projection is 1st-order in time, and that temporal error concentrates in
# a thin layer along the (exact-Dirichlet) boundaries -- a ParaView slice near an edge
# shows u vs u_exact offset by ~1.5% at cfl=0.4, halving as dt halves. The interior and
# the vortex peak amplitudes match to <0.5% regardless. Lower cfl -> tighter boundary match.
solver = IncompressibleSolver(u, v, P, nu=nu, rho=rho, cfl=0.2,
                              implicit_momentum=True, conv_order=2, ncorr=2, n_nonorth=2)

# ---- initial condition = exact field at t=0 ----
cx, cy = cells.center[:, 0], cells.center[:, 1]
u.cell[:] = exact_u(cx, cy, 0.0)
v.cell[:] = exact_v(cx, cy, 0.0)

# ---- boundary faces: feed the exact velocity there each step ----
# PHYSICAL boundary faces only: exclude interior (0) AND partition/halo (10) faces, so
# this stays correct under MPI (halo faces get their velocity from the neighbour rank).
fname = np.asarray(faces.name)
bnd = (fname != 0) & (fname != 10)
fbx, fby = faces.center[bnd, 0], faces.center[bnd, 1]

vol = np.asarray(cells.volume)


def kinetic_energy():
  ke = 0.5 * (np.asarray(u.cell) ** 2 + np.asarray(v.cell) ** 2)
  return COMM.allreduce(float(np.sum(vol * ke)), op=MPI.SUM)


# Divergence of the freshly-sampled exact field (before any step). The FV
# divergence of a cell-sampled smooth div-free field is O(h) (halves under
# refinement), so this is the discretization FLOOR the solver cannot beat --
# the useful check is that the final divergence stays at this floor.
solver.uw[bnd] = exact_u(fbx, fby, 0.0)
solver.vw[bnd] = exact_v(fbx, fby, 0.0)
div0 = solver.divergence_norm()

ke0 = kinetic_energy()
COMM.Barrier()
ts = MPI.Wtime()
if RANK == 0:
  print("Start Taylor-Green (incompressible projection) ...")

def save_vtk(niter, miter, t, d_t):
  ue_c, ve_c = exact_u(cx, cy, t), exact_v(cx, cy, t)
  errmag = np.sqrt((np.asarray(u.cell) - ue_c) ** 2 + (np.asarray(v.cell) - ve_c) ** 2)
  domain.save_on_cell_multi(
      ["u", "v", "P", "u_exact", "v_exact", "err"],
      [u.cell, v.cell, P.cell, ue_c, ve_c, errmag],
      d_t, t, niter, miter)


time = 0.0
niter = 0
miter = 0
output_every = 25
save_vtk(0, miter, 0.0, 0.0); miter += 1     # initial frame
while time < T:
  # impose the exact boundary velocity at the current time level
  solver.uw[bnd] = exact_u(fbx, fby, time)
  solver.vw[bnd] = exact_v(fbx, fby, time)
  d_t = solver.step()
  time += d_t
  niter += 1
  if niter % output_every == 0:
    save_vtk(niter, miter, time, d_t)          # u, v, P + exact + error, for ParaView
    miter += 1

save_vtk(niter, miter, time, solver.dt)        # final frame

te = MPI.Wtime()
walltime = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# ---- errors vs exact ----
ue, ve = exact_u(cx, cy, time), exact_v(cx, cy, time)
eu, ev = np.asarray(u.cell) - ue, np.asarray(v.cell) - ve
num = COMM.allreduce(float(np.sum(vol * (eu * eu + ev * ev))), op=MPI.SUM)
den = COMM.allreduce(float(np.sum(vol * (ue * ue + ve * ve))), op=MPI.SUM)
l2_rel = np.sqrt(num / den)                       # relative L2 velocity error
linf = COMM.allreduce(float(np.max(np.sqrt(eu * eu + ev * ev))), op=MPI.MAX)

ke1 = kinetic_energy()
ke_exact_ratio = np.exp(-4.0 * nu * time)         # KE(t)/KE(0) analytic
div = solver.divergence_norm()
glob_cells = COMM.allreduce(domain.nbcells, op=MPI.SUM)

if RANK == 0:
  print("\n============= Taylor-Green vortex vs EXACT (incompressible) =============")
  print(f"  mesh {nx}x{ny} ({glob_cells} cells)   nu={nu}   iters={niter}   t={time:.4f}   {walltime:.2f}s")
  print(f"  relative L2 velocity error : {l2_rel:.3e}")
  print(f"  Linf velocity error        : {linf:.3e}")
  print(f"  kinetic energy  KE/KE0     : {ke1/ke0:.5f}   exact exp(-4nu t) = {ke_exact_ratio:.5f}"
        f"   (err {abs(ke1/ke0 - ke_exact_ratio):.2e})")
  # NB: divergence_norm is the cell-centred (Green-Gauss) divergence -- a collocated
  # diagnostic, NOT the conserved quantity (the Rhie-Chow FACE flux is divergence-free).
  # The real incompressibility proof is that u/energy track the exact solution above.
  print(f"  cell (Green-Gauss) div     : sampled-exact {div0:.2e}  final {div:.2e}  (collocated diagnostic)")
  print(f"  VTK (u,v,P,u_exact,v_exact,err) -> ./vtk_results/   ({miter} frames, open in ParaView)")
  print("========================================================================\n")
