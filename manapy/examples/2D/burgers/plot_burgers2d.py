#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot exact vs simulated for the 2D viscous Burgers travelling wave.

Runs one simulation, extracts the cell profile along the mid-line, and overlays
the exact tanh solution + the pointwise error. Saves a PNG (serial run).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manapy.api.mesh import Mesh
from manapy.api.models import BurgersModel

# ---- problem ----
uL, uR = 1.0, 0.0
nu = 0.02
x0 = 0.20
T = 0.40
s = 0.5 * (uL + uR)


def exact(x, t):
  return 0.5 * (uL + uR) - 0.5 * (uL - uR) * np.tanh((uL - uR) * (x - x0 - s * t) / (4.0 * nu))


# ---- solve (api) ----
nx, ny = 200, 8
mesh = Mesh.rectangle(bounds=((0.0, 1.0), (0.0, 0.2)), n=(nx, ny), cell_type="triangle")
u = mesh.field(
    "u",
    init=lambda x, y, z: exact(x, 0.0),
    bc={"in": ("dirichlet", exact(0.0, 0.0)), "out": ("dirichlet", exact(1.0, 0.0)),
        "upper": "neumann", "bottom": "neumann"},
    limiter="vanalbada",
)
BurgersModel(u, mesh, nu=nu, order=2, cfl=0.4, scheme="rusanov").run(
    T, output_every=10**9, output_mode="cell")

# ---- extract mid-line profile ----
c = np.asarray(mesh.domain.cells.center)
uc = np.asarray(u.cell)
strip = np.abs(c[:, 1] - 0.10) < 0.03
xs = c[strip, 0]
un = uc[strip]
o = np.argsort(xs)
xs, un = xs[o], un[o]

ue = exact(xs, T)
err = un - ue
xf = np.linspace(0, 1, 1000)

l2 = np.sqrt(np.mean((un - ue) ** 2))
linf = np.max(np.abs(err))

# ---- plot ----
fig, (ax, axe) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

ax.plot(xf, exact(xf, 0.0), "--", color="0.6", lw=1.5, label="exact  t = 0 (IC)")
ax.plot(xf, exact(xf, T), "-", color="#111111", lw=2.2, label=f"exact  t = {T}")
ax.plot(xs, un, "o", ms=4.5, mfc="#d64545", mec="#7a1f1f", mew=0.5, alpha=0.9,
        label=f"simulé  t = {T}  (order 2, Rusanov)")
ax.set_ylabel("u")
ax.set_title(f"Burgers 2D visqueux — exact vs simulé   (ν={nu}, s={s}, {nx}×{ny})")
ax.legend(loc="upper right", frameon=False)
ax.grid(True, alpha=0.25)
ax.text(0.03, 0.12, f"L2 = {l2:.2e}\nL∞ = {linf:.2e}", transform=ax.transAxes,
        va="bottom", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))

axe.axhline(0, color="0.6", lw=0.8)
axe.plot(xs, err, "-", color="#2b6cb0", lw=1.5)
axe.fill_between(xs, err, color="#2b6cb0", alpha=0.2)
axe.set_ylabel("u_num − u_exact")
axe.set_xlabel("x")
axe.grid(True, alpha=0.25)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.realpath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "burgers2d_exact_vs_sim.png")
fig.savefig(path, dpi=140)
print("saved:", path)
