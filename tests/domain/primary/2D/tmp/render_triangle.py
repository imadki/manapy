import tkinter as tk

import meshio
import os
import numpy as np
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.base.base import Struct

dim = 2
float_precision = 'float32'
PATH = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(PATH, 'mesh', 'triangles.msh')
#PATH = os.path.join(PATH, 'mesh', 'cuboid.msh')

running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)
mesh_partition = MeshPartition(PATH, dim=dim, conf=running_conf, periodic=[0, 0, 0])
domain = Domain(dim=dim, conf=running_conf)

size = 5

d_cells = []
d_points = []
d_center = []
d_volume = []
d_cellfid = []
d_cellnid = []
d_halonid = []
d_loctoglob = []
d_node_loctoglob = []

for i in range(size):
  domain._create_domain(size, i)
  d_cells.append(domain._cells._nodeid)
  d_points.append(domain._nodes._vertex)
  d_center.append(domain._cells._center)
  d_volume.append(domain._cells._volume)
  d_halonid.append(domain._cells._halonid)
  d_loctoglob.append(domain._cells._loctoglob)
  d_cellfid.append(domain._cells._cellfid)
  d_cellnid.append(domain._cells._cellnid)
  d_node_loctoglob.append(domain._nodes._loctoglob)





# Create the window
root = tk.Tk()
root.title("Tkinter Window")

# Create a canvas
canvas = tk.Canvas(root, width=1000, height=1000)
canvas.pack()

colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta", "black", "gray"]

for k in range(size):
  ld_cells = d_cells[k]
  ld_points = d_points[k]
  for i in range(len(ld_cells)):
    cell = ld_cells[i]
    point_ids = cell[0:cell[-1]]
    # for pid in point_ids:
    #   p = ld_points[pid]
    #   p = p * 50 + 40
    #   canvas.create_text(p[0], p[1], text=str(pid), font=("Arial", 10), fill="blue")
    p = d_center[k][i] * 60 + 50
    canvas.create_text(p[0], p[1], text=str(i), font=("Arial", 10), fill=colors[k % len(colors)])
    ori_i = d_loctoglob[k][i]
    canvas.create_text(p[0], p[1] + 500, text=str(ori_i), font=("Arial", 10), fill=colors[k % len(colors)])

root.mainloop()
