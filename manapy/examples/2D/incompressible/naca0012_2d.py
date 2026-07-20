#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incompressible flow around a NACA 0012 airfoil (external flow on an unstructured mesh).

A realistic curved-boundary test for the projection solver + the deferred non-orthogonal
correction (n_nonorth): the mesh is unstructured triangles refined near the airfoil, so the
faces are genuinely skewed (unlike a structured mesh). At Re=500, AoA=4deg the flow is laminar
and near-steady; the expected features are a leading-edge stagnation point (high p), a suction
peak on the upper surface (high |u|, low p), and a thin no-slip boundary layer.

manapy only has 4 boundary patches, so we map them for external flow:
    in(1)=left inflow, out(2)=right outflow, upper(3)=top AND bottom farfield (freestream),
    bottom(4)=the airfoil (no-slip). Net boundary flux is zero (freestream is divergence-free
    and the airfoil is closed), so the pure-Neumann pressure Poisson stays compatible.
Note: the solver has only Dirichlet-velocity boundaries (no true convective outflow), so the
outflow u=U_inf is an approximation -- fine for laminar attached flow, crude if the wake is strong.

Run:
    python3 naca0012_2d.py            # serial (recommended: mesh is generated once)
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.api.mesh import Mesh
from manapy.api import meshgen
from manapy.solvers.incompressible.system import IncompressibleSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ---- parameters ----
U, chord, Re = 1.0, 1.0, 500.0
nu = U * chord / Re
alpha = np.deg2rad(4.0)
u_inf, v_inf = U * np.cos(alpha), U * np.sin(alpha)
MESH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "naca0012.msh")


def naca0012_thickness(x):
  """Half-thickness of a NACA 0012 (closed trailing edge)."""
  return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def build_mesh(path):
  """Write a NACA 0012 (chord [0,1]) in a rectangular farfield.

  HYBRID airfoil mesh (the CFD-standard topology): a STRUCTURED boundary-layer of quads
  hugging the airfoil (wall-aligned -> resolves the boundary layer AND is nearly orthogonal
  to the wall, which is where accuracy matters most), then unstructured triangles fill the
  farfield. The non-orthogonality then lives in the triangle region / the BL-to-triangle
  transition, where the deferred non-ortho correction (n_nonorth) handles it.
  """
  n = 100
  xc = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))        # cosine clustering at LE/TE
  yt = naca0012_thickness(xc)
  pts = ([(xc[i], yt[i]) for i in range(n - 1, -1, -1)]        # upper TE -> LE
         + [(xc[i], -yt[i]) for i in range(1, n)])             # lower LE -> TE
  g, pid, aids = [], 1, []
  for (x, y) in pts:
    g.append(f"Point({pid}) = {{{x:.6f}, {y:.6f}, 0, 0.02}};"); aids.append(pid); pid += 1
  g.append(f"Spline(1) = {{{','.join(map(str, aids))},{aids[0]}}};")
  g.append("Line Loop(1) = {1};")
  b, (xL, xR, yF) = pid, (-6.0, 14.0, 7.0)
  g += [f"Point({b}) = {{{xL},{-yF},0,1.5}};", f"Point({b+1}) = {{{xR},{-yF},0,1.5}};",
        f"Point({b+2}) = {{{xR},{yF},0,1.5}};", f"Point({b+3}) = {{{xL},{yF},0,1.5}};",
        f"Line(10) = {{{b},{b+1}}};", f"Line(11) = {{{b+1},{b+2}}};",
        f"Line(12) = {{{b+2},{b+3}}};", f"Line(13) = {{{b+3},{b}}};",
        "Line Loop(2) = {10,11,12,13};", "Plane Surface(1) = {2,1};",
        # farfield triangle sizing: fine near the airfoil, coarse far away
        "Field[1]=Distance; Field[1].CurvesList={1};",
        "Field[2]=Threshold; Field[2].InField=1; Field[2].SizeMin=0.02; Field[2].SizeMax=1.5;"
        " Field[2].DistMin=0.1; Field[2].DistMax=4.0;", "Background Field = 2;",
        # STRUCTURED boundary layer (quads) grown from the airfoil surface
        "Field[3]=BoundaryLayer; Field[3].CurvesList={1}; Field[3].Size=0.0015;"
        " Field[3].Ratio=1.3; Field[3].Thickness=0.05; Field[3].Quads=1;",
        "BoundaryLayer Field = 3;",
        'Physical Curve("in", 1)     = {13};',        # left  -> inflow
        'Physical Curve("out", 2)    = {11};',        # right -> outflow
        'Physical Curve("upper", 3)  = {10, 12};',    # top+bottom -> freestream
        'Physical Curve("bottom", 4) = {1};',         # airfoil -> no-slip
        'Physical Surface("domain", 1) = {1};']
  meshgen._run_gmsh("\n".join(g), 2, path)


# ---- mesh: generate once (rank 0), everyone loads the same file (MPI-safe) ----
if RANK == 0 and not os.path.exists(MESH):
  build_mesh(MESH)
COMM.Barrier()

mesh = Mesh(MESH, dim=2)
domain = mesh.domain
cells = domain.cells

u = mesh.field("u")
v = mesh.field("v")
# pressure: Neumann everywhere except a Dirichlet reference at the outflow.
P = mesh.field("P", bc={"in": "neumann", "upper": "neumann", "bottom": "neumann",
                        "out": ("dirichlet", 0.0)})

solver = IncompressibleSolver(
    u, v, P, nu=nu, rho=1.0, cfl=0.4, implicit_momentum=True, conv_order=2, ncorr=2,
    n_nonorth=2,                                     # deferred non-ortho correction (skew mesh)
    u_bc={"in": u_inf, "out": u_inf, "upper": u_inf, "bottom": 0.0},
    v_bc={"in": v_inf, "out": v_inf, "upper": v_inf, "bottom": 0.0})

u.cell[:] = u_inf
v.cell[:] = v_inf


def save_vtk(niter, miter):
  uc, vc = np.asarray(u.cell), np.asarray(v.cell)
  spd = np.sqrt(uc ** 2 + vc ** 2)
  domain.save_on_cell_multi(["u", "v", "P", "speed"], [u.cell, v.cell, P.cell, spd],
                            solver.dt, 0.0, niter, miter)


ts = MPI.Wtime()
if RANK == 0:
  print(f"NACA 0012  Re={Re:.0f}  AoA=4deg  (nu={nu:.4f}), non-ortho correction ON ...")

uold = np.asarray(u.cell).copy()
miter = 0
nmax = int(os.environ.get("NSTEP", 4000))
for it in range(nmax):
  solver.step()
  if it % 250 == 0:
    ch = COMM.allreduce(float(np.max(np.abs(np.asarray(u.cell) - uold))), op=MPI.MAX)
    uold = np.asarray(u.cell).copy()
    save_vtk(it, miter); miter += 1
    if RANK == 0:
      umax = COMM.allreduce(float(np.max(np.sqrt(u.cell ** 2 + v.cell ** 2))), op=MPI.MAX)
      print(f"  it {it:5d}  |u|max={umax:.3f}  div={solver.divergence_norm():.2e}  dU={ch:.2e}", flush=True)
    if it > 0 and ch < 2e-4:
      if RANK == 0:
        print("  steady")
      break

save_vtk(nmax, miter)
walltime = COMM.reduce(MPI.Wtime() - ts, op=MPI.MAX, root=0)
if RANK == 0:
  umax = COMM.allreduce(float(np.max(np.sqrt(u.cell ** 2 + v.cell ** 2))), op=MPI.MAX)
  print(f"\n  |u|max={umax:.3f}  walltime={walltime:.1f}s")
  print(f"  VTK (u, v, P, speed) -> ./vtk_results/   (open in ParaView; airfoil is the hole)")
