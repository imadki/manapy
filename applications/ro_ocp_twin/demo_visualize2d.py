#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient cross-flow RO demo with VTK output for ParaView.

Runs the manapy RO solver (resolved concentration polarization + transient
fouling) on the wall-graded channel mesh and writes a VTK time series to
``./vtk_results/`` (one frame per save).  Open ``vtk_results/*.pvtu`` in
ParaView and play the animation to watch the salt boundary layer build at the
membrane while the fouling layer makes the permeate flux decline.

Run from this directory so the frames land in ``ro_ocp_twin/vtk_results/``:
    python3 demo_visualize2d.py
Fields saved (cell data): ``salt`` [g/L], ``u``/``v`` [m/s], ``cp_modulus`` (c/feed).
"""
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.ro import ReverseOsmosisSolver

HERE = os.path.dirname(os.path.realpath(__file__))
MESH = os.path.join(HERE, "..", "..", "meshes", "ro_channel_graded.msh")
FEED = 35.0

# Creating the domain also (re)initializes ./vtk_results/ (rank 0 wipes it).
dom = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)

c = Variable(domain=dom, BC={"in": "dirichlet", "out": "neumann",
                             "upper": "neumann", "bottom": "neumann"},
             values_dict={"in": FEED}, name="salt")
u = Variable(domain=dom, name="u")
v = Variable(domain=dom, name="v")
c.cell[:] = FEED

s = ReverseOsmosisSolver(c, vel=(u, v), feed_conc=FEED, U0=0.05, D=1.0e-7,
                         A_w=4.2e-12, B_s=3.5e-8, dP=6.0e6, osmotic_coeff=8.0e4,
                         fouling=True, fouling_coeff=0.5, flow_model="crossflow",
                         velocity_profile="parabolic", order=2, scheme="upwind")

N_FRAMES, STEPS_PER_FRAME = 12, 120
print(f"Writing {N_FRAMES} VTK frames to {os.path.join(HERE, 'vtk_results')} ...")


def save_frame(miter):
    cp_mod = c.cell / FEED                       # local polarization c/c_feed
    dom.save_on_cell_multi(["salt", "u", "v", "cp_modulus"],
                           [c.cell, u.cell, v.cell, cp_mod],
                           dt=0.0, time=s.time, niter=s.niter, miter=miter)


save_frame(0)                                    # initial state
for frame in range(1, N_FRAMES + 1):
    s.run(nsteps=STEPS_PER_FRAME)
    save_frame(frame)
    d = s.diagnostics()
    print(f"  frame {frame:2d}  t={d['time']:.3f}s  flux={d['flux_LMH']:6.2f} LMH  "
          f"cw_mean={d['cw_mean']:6.2f}  Rf/Rm={d['Rf_over_Rm']:.3f}")

print("Done. Open in ParaView:")
print(f"    paraview {os.path.join(HERE, 'vtk_results')}/*.pvtu")
print("Then press Play; colour by 'salt' (or 'cp_modulus') and zoom on the membrane (y=0).")
