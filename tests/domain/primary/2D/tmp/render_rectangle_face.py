import tkinter as tk


import sys
import os

# Add the parent directory to sys.path
sys.path.append("/media/aben-ham/SSD/aben-ham/work/manapy/tests/domain/primary/2D")

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
canvas = tk.Canvas(root, width=3500, height=3000)
canvas.pack()

colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta", "black", "gray"]
SIZE = 0

def ft_put_item(x, y, item, colorId, fontSize, yOffset):
  # 0 => 4k
  if fontSize == -1:
    fontSize = 12
    if SIZE == 0:
      fontSize = 24
  if SIZE == 0:
    canvas.create_text(x * 100 + 200, y * 100 + yOffset * 2, text=str(item), font=("Arial", fontSize),
                       fill=colors[(colorId + 1) % len(colors)])
  else:
    canvas.create_text(x * 50 + 100, y * 50 + yOffset, text=str(item), font=("Arial", fontSize),
                       fill=colors[(colorId + 1) % len(colors)])

# =====================

def show_node():
  for k in range(size):
    d_nodes = domain_tables.d_nodes[k]
    d_node_name = domain_tables.d_node_name[k]
    d_node_oldname = domain_tables.d_node_oldname[k]

    for i in range(len(d_nodes)):
      p = d_nodes[i]
      ft_put_item(p[0], p[1], d_node_name[i], k, -1, 400)
      ft_put_item(p[0], p[1], d_node_oldname[i], k, -1, 800)

# ===================
def show_partition():
  for k in range(size):
    cells = domain_tables.d_cells[k]
    for i in range(len(cells)):
      p = domain_tables.d_cell_center[k][i]
      ft_put_item(p[0], p[1], k, k, -1, 400)


# show_node()
show_partition()
root.mainloop()
