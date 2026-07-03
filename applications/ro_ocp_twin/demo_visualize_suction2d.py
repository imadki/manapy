#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dramatic concentration-polarization demo (pure suction) with VTK output.

Uniform wall-ward suction (no cross-flow to sweep the salt away) so the
polarization fills the whole channel: the wall reaches ~2.2x the feed and the
resolved profile matches film theory exp(Jw*H/D).  Writes a VTK time series so
you can watch the layer BUILD UP in ParaView.

Output: ``./vtk_results_suction/`` (this script moves the writer's default
``vtk_results/`` there at the end, so it does NOT clobber the cross-flow demo's
``vtk_results/``).  Run from this directory:
    python3 demo_visualize_suction2d.py
Open ``vtk_results_suction/visu..pvtu`` in ParaView, colour by ``salt`` and Play.
"""
import os
import shutil
import numpy as np
from mpi4py import MPI

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.ro import ReverseOsmosisSolver

HERE = os.path.dirname(os.path.realpath(__file__))
MESH = os.path.join(HERE, "..", "..", "meshes", "ro_channel_quad.msh")
OUT = os.path.join(HERE, "vtk_results_suction")
FEED, D = 35.0, 1.0e-6

dom = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)
# pure suction: top held at feed (bulk), membrane sucks water down, no cross-flow
c = Variable(domain=dom, BC={"in": "neumann", "out": "neumann",
                             "upper": "dirichlet", "bottom": "neumann"},
             values_dict={"upper": FEED}, name="salt")
u, v = Variable(domain=dom), Variable(domain=dom)
c.cell[:] = FEED
s = ReverseOsmosisSolver(c, vel=(u, v), feed_conc=FEED, U0=0.0, D=D,
                         A_w=4.2e-12, B_s=5.0e-8, dP=2.5e7, osmotic_coeff=8.0e4,
                         fouling=False, flow_model="uniform_suction",
                         order=2, scheme="upwind")

N_FRAMES, STEPS_PER_FRAME = 16, 400
print(f"Building the polarization layer over {N_FRAMES} VTK frames ...")


def save_frame(miter):
    dom.save_on_cell_multi(["salt", "cp_modulus"], [c.cell, c.cell / FEED],
                           dt=0.0, time=s.time, niter=s.niter, miter=miter)


save_frame(0)
for frame in range(1, N_FRAMES + 1):
    s.run(nsteps=STEPS_PER_FRAME)
    save_frame(frame)
    cw, Jw, cp = s._membrane_state()
    print(f"  frame {frame:2d}  t={s.time:6.1f}s  c_wall={float(cw.mean()):5.1f} g/L "
          f"(={float(cw.mean())/FEED:.2f}x feed)")

# move the writer's default output to a dedicated folder (rank 0), keeping the
# cross-flow demo's vtk_results/ intact.
if MPI.COMM_WORLD.Get_rank() == 0:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    if os.path.isdir("vtk_results"):
        shutil.move("vtk_results", OUT)
    print(f"Done -> {OUT}")
    print(f"    paraview {OUT}/visu..pvtu    (colour by 'salt' or 'cp_modulus', press Play)")
