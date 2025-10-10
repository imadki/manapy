import tkinter as tk


import sys
import os

# Add the parent directory to sys.path

sys.path.append("/home/aben-ham/Desktop/work/manapy/tests/domain/primary/")

import subprocess
import os
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'domain'))
from create_domain import Mesh, Partitioning, LocalDomain
from local_domain_1cpu_testing import LocalDomain1Cpu

mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[0] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh



mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh, float_precision)
size = 4
local_domains = partitioning.create_sub_domains(nb_parts=size)



# Create the window
root = tk.Tk()
root.title("Tkinter Window")

# Create a canvas
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(root, width=3500, height=3000)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)

scrollregion_width = 3500
scrollregion_height = 3000  # You can adjust this value as per the content size


def on_canvas_configure(event, canvas, scrollregion_width, scrollregion_height):
  canvas.configure(scrollregion=(0, 0, scrollregion_width, scrollregion_height))

# Bind a callback to dynamically set the scroll region based on content size
canvas.bind("<Configure>", lambda event: on_canvas_configure(event, canvas, scrollregion_width, scrollregion_height))


colors = ["red", "blue", "green", "orange", "purple", "yellow", "magenta", "black", "gray", "white", "black"]
SIZE = 1

def draw_rec(center_x, center_y, width, height, k, yOffset):
  # Calculate the coordinates of the top-left and bottom-right corners
  center_x = center_x * 50 + 100
  center_y = center_y * 50 + yOffset
  width *= 50
  height *= 50
  top_left_x = center_x - width // 2
  top_left_y = center_y - height // 2
  bottom_right_x = center_x + width // 2
  bottom_right_y = center_y + height // 2

  # Draw the rectangle
  color = colors[k]
  canvas.create_rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y, fill=color)

def ft_put_item(x, y, item, colorId, fontSize, yOffset):
  # 0 => 4k
  if fontSize == -1:
    fontSize = 12
    if SIZE == 0:
      fontSize = 24
  if SIZE == 0:
    canvas.create_text(x * 100 + 200, y * 100 + yOffset * 2, text=str(item), font=("Arial", fontSize),
                       fill=colors[colorId % (len(colors)-2)])
  else:
    canvas.create_text(x * 50 + 100, y * 50 + yOffset, text=str(item), font=("Arial", fontSize),
                       fill=colors[colorId % (len(colors)-2)])

# =====================

def get_cell_center(cell_nodes, nodes):
  nb = cell_nodes[-1]
  v = nodes[cell_nodes[0:nb]]
  return np.sum(v, axis=0) / 4.0

def show_node():
  for k in range(size):
    d_nodes = local_domains[k].nodes
    d_cells = local_domains[k].cells
    d_cell_loctoglob = local_domains[k].cell_loctoglob
    d_node_loctoglob = local_domains[k].node_loctoglob



    for i in range(len(d_cells)):
      p = get_cell_center(d_cells[i], d_nodes) * 2
      draw_rec(p[0], p[1], 1, 1, k, 100)


    for i in range(len(d_cells)):
      p = get_cell_center(d_cells[i], d_nodes) * 2
      draw_rec(p[0], p[1], 0.6, 0.4, 10, 100)
      cellid = d_cell_loctoglob[i]
      ft_put_item(p[0], p[1], f"{cellid},{k}", 9, -1, 100)

      cell = d_cells[i]
      for j in range(cell[-1]):
        node_id = cell[j]
        gid = d_node_loctoglob[node_id]
        p = d_nodes[node_id] * 2
        ft_put_item(p[0], p[1], gid, 9, -1, 100)




#show_face()
show_node()
# show_partition()
root.mainloop()
