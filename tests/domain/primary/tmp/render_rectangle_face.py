import tkinter as tk


import sys
import os

# Add the parent directory to sys.path

sys.path.append("/home/aben-ham/Desktop/work/manapy/tests/domain/primary/2D")

import subprocess
import os
import numpy as np
from helpers.DomainTables import DomainTables

mpi_exec = "/usr/bin/mpirun"
python_exec = "/home/aben-ham/anaconda3/envs/work/bin/python3"
float_precision = 'float32'
dim=2
mesh_name = 'rectangles.msh'




def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.getcwd()
  mesh_file_path = os.path.join(root_file, '..', 'mesh', mesh_name)
  script_path = os.path.join(root_file, '..', 'helpers', 'create_partitions_mpi_worker.py')
  cmd = [mpi_exec, "-n", str(nb_partitions), "--oversubscribe", python_exec, script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)

domain_tables = DomainTables(nb_partitions=7, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
size = domain_tables.nb_partitions

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
SIZE = 0

def draw_rec(center_x, center_y, width, height, k, yOffset):
  # Calculate the coordinates of the top-left and bottom-right corners
  center_x = center_x * 100 + 200
  center_y = center_y * 100 + yOffset
  width *= 100
  height *= 100
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

def show_node():
  for k in range(size):
    d_nodes = domain_tables.d_nodes[k]
    d_node_name = domain_tables.d_node_name[k]
    d_node_oldname = domain_tables.d_node_oldname[k]
    d_face_center = domain_tables.d_face_center[k]
    d_face_name = domain_tables.d_face_name[k]
    d_cell_center = domain_tables.d_cell_center[k]



    # for i in range(len(d_cell_center)):
    #   p = d_cell_center[i] * 2
    #   draw_rec(p[0], p[1], 2, 1, k, 50)


    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 2, 1, k, 600)



    for i in range(len(d_nodes)):
      p = d_nodes[i] * 2
      draw_rec(p[0], p[1], 0.5, 0.5, 10, 600)
      ft_put_item(p[0], p[1], d_node_name[i], 9, -1, 600)

    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 2, 1, k, 50)

    for i in range(len(d_face_center)):
      p = d_face_center[i] * 2
      draw_rec(p[0], p[1], 0.5, 0.5, 10, 50)
      ft_put_item(p[0], p[1], d_face_name[i], 9, -1, 50)

    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 0.4, 0.4, 10, 50)
      ft_put_item(p[0], p[1], k, 9, -1, 50)

# ===================
def show_partition():
  for k in range(size):
    cells = domain_tables.d_cells[k]
    d_cell_center = domain_tables.d_cell_center[k]

    for i in range(len(cells)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 2, 1, k, 100)

    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 0.5, 0.5, 10, 100)
      ft_put_item(p[0], p[1], k, 9, -1, 50)

def show_face():
  for k in range(size):
    d_nodes = domain_tables.d_nodes[k]
    d_node_name = domain_tables.d_node_name[k]
    d_node_oldname = domain_tables.d_node_oldname[k]
    d_face_center = domain_tables.d_face_center[k]
    d_face_name = domain_tables.d_face_name[k]
    d_face_oldname = domain_tables.d_face_oldname[k]
    d_cell_center = domain_tables.d_cell_center[k]


    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 2, 1, k, 600)

    for i in range(len(d_face_name)):
      p = d_face_center[i] * 2
      draw_rec(p[0], p[1], 0.5, 0.5, 10, 600)
      ft_put_item(p[0], p[1], d_face_name[i], 9, -1, 600)

    for i in range(len(d_cell_center)):
      p = d_cell_center[i] * 2
      draw_rec(p[0], p[1], 2, 1, k, 50)

    for i in range(len(d_face_name)):
      p = d_face_center[i] * 2
      draw_rec(p[0], p[1], 0.5, 0.5, 10, 50)
      ft_put_item(p[0], p[1], d_face_oldname[i], 9, -1, 50)


#show_face()
# show_node()
show_partition()
root.mainloop()
