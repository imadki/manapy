import tkinter as tk
import sys
import subprocess
import os
import numpy as np

# Add the parent directory to sys.path
helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(helpers_path)
from helpers.DomainTables import DomainTables

mpi_exec = "/usr/bin/mpirun"
python_exec = "/home/aben-ham/anaconda3/envs/work/bin/python3"
float_precision = 'float32'
dim=2
mesh_name = 'triangles.msh'




def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.getcwd()
  mesh_file_path = os.path.join(root_file, '..', 'mesh', mesh_name)
  script_path = os.path.join(root_file, '..', 'helpers', 'create_partitions_mpi_worker.py')
  cmd = [mpi_exec, "-n", str(nb_partitions), "--oversubscribe", python_exec, script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)

domain_tables = DomainTables(nb_partitions=5, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
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


def getColor(i, flag=1):
  colors = ["red", "blue", "green", "orange", "purple", "yellow", "magenta", "black", "gray",
            "white", "black"]
  if flag == 1:
    return colors[i % (len(colors)-2)]
  return colors[i % 2 + 9]

def scale(p):
  p[:] = p * 100 + 200

def create_polygon(points, color):
  canvas.create_polygon(list(points), fill=color, outline="black", width=1)

def get_rect_point(center, x, y):
  return np.array([
    center[0] - x,
    center[1] - y,
    center[0] + x,
    center[1] - y,
    center[0] + x,
    center[1] + y,
    center[0] - x,
    center[1] + y,
  ])

def ft_put_item(p, item, color, fontSize):
  canvas.create_text(p[0], p[1], text=str(item), font=("Arial", fontSize),
                     fill=color)

def test():
  for k in range(size):
    d_cell_nodeid = domain_tables.d_cell_nodeid[k]
    d_nodes = domain_tables.d_nodes[k]
    d_node_name = domain_tables.d_node_name[k]
    d_node_oldname = domain_tables.d_node_oldname[k]
    d_face_center = domain_tables.d_face_center[k]
    d_face_name = domain_tables.d_face_name[k]
    d_face_oldname = domain_tables.d_face_oldname[k]
    d_cell_center = domain_tables.d_cell_center[k]
    d_cell_loctoglob = domain_tables.d_cell_loctoglob[k]

    for i in range(len(d_cell_center)):
      cell_nodeid = d_cell_nodeid[i][0:d_cell_nodeid[i, -1]]
      p = d_nodes[cell_nodeid][:, 0:2].flatten()
      scale(p)
      create_polygon(p, getColor(k))

    for i in range(len(d_node_name)):
      node_coords = d_nodes[i]
      p = get_rect_point(node_coords, 0.1, 0.1)
      scale(p)
      create_polygon(p, getColor(1, flag=0))

      p = node_coords
      scale(p)
      ft_put_item(p, d_node_oldname[i], getColor(0, flag=0), 12)

    for i in range(len(d_cell_center)):
      g_index = d_cell_loctoglob[i]
      p = d_cell_center[i]
      scale(p)
      ft_put_item(p, f"{k}, {g_index}", getColor(0, flag=0), 12)

test()
root.mainloop()
