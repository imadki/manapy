#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense brine plume via the high-level MultilayerSWModel API (3-line launch).

Same plunging-plume case as brine_plume_hllc2d.py, but driven through
manapy.api.models.MultilayerSWModel instead of the low-level solver loop --
demonstrates the high-level entry point. Writes VTK (h1,s1,h2,s2) for ParaView.
"""
import numpy as np
from manapy.api.mesh import Mesh
from manapy.api.models import MultilayerSWModel

NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}

ETA, SLOPE = 0.15, 0.06
HF, HP, XL = 0.004, 0.05, 0.25

mesh = Mesh.rectangle(bounds=((0., 1.5), (0., 0.1)), n=(120, 5), cell_type="triangle")
xc = np.asarray(mesh.domain.cells.center)[:, 0]

Z = mesh.field("Z", init=lambda x, y, z: -SLOPE * x)
h1_0 = HF + (HP - HF) * 0.5 * (1.0 - np.tanh((xc - XL) / 0.05))
h2_0 = (ETA + SLOPE * xc) - h1_0
layers = [
  {'h': mesh.field("h1", init=h1_0, bc=NEU), 'hu': mesh.field("hu1", init=0., bc=NEU),
   'hv': mesh.field("hv1", init=0., bc=NEU), 's': mesh.field("s1", init=h1_0, bc=NEU)},
  {'h': mesh.field("h2", init=h2_0, bc=NEU), 'hu': mesh.field("hu2", init=0., bc=NEU),
   'hv': mesh.field("hv2", init=0., bc=NEU), 's': mesh.field("s2", init=0., bc=NEU)},
]

# the whole run in one call: robust HLLC flux + turbulent entrainment
model = MultilayerSWModel(layers, mesh, rho=[1035., 1000.], Z=Z, scheme="hllc",
                          entrain=True, Mann=0.01, cfl=0.45)
model.run(T=2.0, output_every=200, output_mode="cell")

# report the dilution the API run produced
h1 = np.asarray(layers[0]['h'].cell)
c1 = np.asarray(layers[0]['s'].cell) / np.maximum(h1, 1e-12)
m = h1 > 2 * HF
conc = float(np.sum(np.asarray(layers[0]['s'].cell)[m]) / max(np.sum(h1[m]), 1e-12))
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
  print(f"[model] MultilayerSWModel run OK — dense-current mean concentration = {conc:.4f}")
