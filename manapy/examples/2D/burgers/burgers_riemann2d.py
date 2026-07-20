#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonlinear viscous Burgers (2D), order-2 MUSCL -- built with the high-level api.

    u_t + d/dx(u^2/2) + d/dy(u^2/2) = nu (u_xx + u_yy)

Travelling viscous shock. Viscous Burgers has the EXACT travelling-wave solution

    u(x,t) = (uL+uR)/2 - (uL-uR)/2 * tanh[ (uL-uR)(x - x0 - s t) / (4 nu) ],
    s = (uL+uR)/2  (Rankine-Hugoniot speed).

The data varies in x only, so the y-flux vanishes and it reduces to 1-D Burgers
in x. We initialise with the EXACT profile, and pass `exact=` to run() so the
exact field u_exact is saved next to u in every VTK frame and the L2/Linf error
is reported. (A pure step IC has no closed-form exact solution at finite time --
hence the tanh IC, which IS an exact solution for all t.)
"""
from mpi4py import MPI
import numpy as np

from manapy.api.mesh import Mesh
from manapy.api.models import BurgersModel

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ------------------------------------------------------------------ parameters
uL, uR = 1.0, 0.0          # left / right states
x0 = 0.25                  # initial shock location
nu = 0.01                  # viscosity (nu = 0 -> inviscid)
T = 1.0                    # final time
s_exact = 0.5 * (uL + uR)  # Rankine-Hugoniot shock speed
x_exact = x0 + s_exact * T


def u_exact(x, t):
  """Exact viscous travelling-wave solution."""
  return 0.5 * (uL + uR) - 0.5 * (uL - uR) * np.tanh((uL - uR) * (x - x0 - s_exact * t) / (4.0 * nu))


# ------------------------------------------------------------------ mesh (api)
# Generated on the fly -- no .msh file needed. Refined in x (shock direction).
mesh = Mesh.rectangle(bounds=((0.0, 1.0), (0.0, 1.0)), n=(200, 40), cell_type="triangle")

# ------------------------------------------------------------------ field (api)
u = mesh.field(
    "u",
    init=lambda x, y, z: u_exact(x, 0.0),   # exact profile at t = 0
    bc={"in": ("dirichlet", float(u_exact(0.0, 0.0))),   # x = 0
        "out": ("dirichlet", float(u_exact(1.0, 0.0))),  # x = 1
        "upper": "neumann",                              # y = 1
        "bottom": "neumann"},                            # y = 0
    limiter="vanalbada",
)

# ------------------------------------------------------------------ solve (api)
# exact=... maintains an "u_exact" field, saves it next to "u" in every VTK frame,
# and stores the final L2/Linf error on the model.
model = BurgersModel(u, mesh, nu=nu, order=2, cfl=0.4, scheme="rusanov",
                     output=[(u, "u")])
ts = MPI.Wtime()
model.run(T, output_every=500, output_mode="cell",
          exact=lambda x, y, z, t: u_exact(x, t))
te = MPI.Wtime()

# ------------------------------------------------------------------ verify
# (a) full-profile error is already computed by run() -> model.l2_error / linf_error
# (b) shock front position vs the analytic Rankine-Hugoniot location.
u.update_halo_value()
u.update_ghost_value()
c = np.asarray(mesh.domain.cells.center)
uc = np.asarray(u.cell)

strip = np.abs(c[:, 1] - 0.5) < 0.06
xs = c[strip, 0]
us = uc[strip]

umin = COMM.allreduce(uc.min(), op=MPI.MIN)
umax = COMM.allreduce(uc.max(), op=MPI.MAX)
mid = 0.5 * (uL + uR)
local_front = xs[us > mid].max() if np.any(us > mid) else -1.0
front = COMM.allreduce(local_front, op=MPI.MAX)

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("\n================= Burgers 2D (viscous, order 2) =================")
  print(f"  wall time                 : {tt:.3f} s")
  print(f"  solution range  [min,max] : [{umin:.4f}, {umax:.4f}]   (IC states {uR} .. {uL})")
  overshoot = max(umax - uL, uR - umin, 0.0)
  print(f"  monotonicity (overshoot)  : {overshoot:.2e}   (should be ~0)")
  print(f"  vs EXACT profile          : L2 = {model.l2_error:.3e}   Linf = {model.linf_error:.3e}")
  print(f"  shock front  numeric      : x = {front:.3f}")
  print(f"  shock front  Rankine-Hug. : x = {x_exact:.3f}  (s = {s_exact})")
  print(f"  front position error      : {abs(front - x_exact):.3f}")
  print("  (VTK output now contains both 'u' and 'u_exact' per frame)")
  print("=================================================================\n")
