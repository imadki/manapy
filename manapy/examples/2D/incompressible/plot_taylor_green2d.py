#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot exact vs simulated for the incompressible Taylor-Green vortex.

Runs the projection solver, then draws speed(exact) / speed(sim) / error contours
and a centreline velocity profile. Saves a PNG (serial).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from manapy.api.mesh import Mesh
from manapy.solvers.incompressible.system import IncompressibleSolver

Lx = 2.0 * np.pi
nu = 0.05
T = 1.5
nx = 64
CELL = "triangle"


def eu(x, y, t):
  return np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)


def ev(x, y, t):
  return -np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)


mesh = Mesh.rectangle(bounds=((0.0, Lx), (0.0, Lx)), n=(nx, nx), cell_type=CELL)
dom = mesh.domain
cells, faces = dom.cells, dom.faces
u = mesh.field("u"); v = mesh.field("v")
P = mesh.field("P", bc={"in": "neumann", "out": "neumann", "upper": "neumann",
                        "bottom": ("dirichlet", 0.0)})
solver = IncompressibleSolver(u, v, P, nu=nu, rho=1.0, cfl=0.4,
                              implicit_momentum=True, conv_order=2, ncorr=2, n_nonorth=2)

cx, cy = cells.center[:, 0], cells.center[:, 1]
u.cell[:] = eu(cx, cy, 0.0); v.cell[:] = ev(cx, cy, 0.0)
fname = np.asarray(faces.name); bnd = fname != 0
fbx, fby = faces.center[bnd, 0], faces.center[bnd, 1]

t = 0.0
while t < T:
  solver.uw[bnd] = eu(fbx, fby, t)
  solver.vw[bnd] = ev(fbx, fby, t)
  t += solver.step()

un, vn = np.asarray(u.cell), np.asarray(v.cell)
ue, ve = eu(cx, cy, t), ev(cx, cy, t)
sp_num = np.sqrt(un ** 2 + vn ** 2)
sp_exa = np.sqrt(ue ** 2 + ve ** 2)
err = np.sqrt((un - ue) ** 2 + (vn - ve) ** 2)

tri = Triangulation(cx, cy)
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
lv = np.linspace(0, sp_exa.max(), 21)
for a, data, title, cm, lvl in (
    (ax[0], sp_exa, f"|u| exact  (t={t:.2f})", "viridis", lv),
    (ax[1], sp_num, "|u| simulated", "viridis", lv),
    (ax[2], err, "|u_num - u_exact|", "magma", None)):
  tc = a.tricontourf(tri, data, levels=lvl if lvl is not None else 21, cmap=cm)
  fig.colorbar(tc, ax=a, shrink=0.85)
  a.set_aspect("equal"); a.set_title(title); a.set_xlabel("x"); a.set_ylabel("y")

# centreline profile: u along y ~ pi/4. Thin band so y is nearly constant, AND the
# exact reference is evaluated at each cell's OWN (x, y) -- otherwise cells at slightly
# different y (u = sin x * cos y) look artificially "offset" from a single-y curve.
y0 = np.pi / 4
band = np.abs(cy - y0) < 0.35 * (Lx / nx)
o = np.argsort(cx[band])
xs, ys = cx[band][o], cy[band][o]
ax[3].plot(xs, eu(xs, ys, t), "k-", lw=2, label="exact (per-cell y)")
ax[3].plot(xs, un[band][o], "o", ms=4, mfc="#d64545", mec="none", label="simulated")
ax[3].set_title(f"u along y~{y0:.2f}  (thin band)"); ax[3].set_xlabel("x"); ax[3].set_ylabel("u")
ax[3].legend(frameon=False); ax[3].grid(alpha=0.25)

l2 = np.sqrt(np.sum(np.asarray(cells.volume) * ((un - ue) ** 2 + (vn - ve) ** 2)) /
             np.sum(np.asarray(cells.volume) * (ue ** 2 + ve ** 2)))
fig.suptitle(f"Incompressible Taylor-Green vortex — exact vs simulated  "
             f"(nu={nu}, {nx}x{nx} {CELL}, rel-L2={l2:.2e})", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.realpath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "taylor_green_exact_vs_sim.png")
fig.savefig(path, dpi=130)
print("saved:", path)
