import tkinter as tk
import sys
import subprocess
import os
import numpy as np

# Add the parent directory to sys.path
helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(helpers_path)
from create_domain import Domain



float_precision = 'float32'
dim=2
mesh_name = 'triangles.msh'
root_file = os.getcwd()
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_name)
size = 4


domain = Domain(mesh_path, dim, float_precision)

cell_cellnid = domain.cell_cellfid
connectivity = []
for item in cell_cellnid:
  c = item[0:item[-1]]
  c = c[c != -1]
  connectivity.append(c)

options = pymetis.Options()
# options.__setattr__("contig", 1)
a, b = pymetis.part_graph(nparts=16, adjacency=connectivity, options=options)
print(a, np.unique(np.array(b)))

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
  cells = domain.cells
  cell_centers = domain.cell_center
  nodes = domain.nodes
  for i in range(len(cells)):
    cn = cells[i, 0:cells[i, -1]]
    p = nodes[cn][:, 0:2].flatten()
    scale(p)
    create_polygon(p, getColor(b[i]))

  for i in range(len(cells)):
    center = cell_centers[i]
    p = get_rect_point(center, 0.1, 0.1)
    scale(p)
    create_polygon(p, getColor(1, flag=0))

    p = center
    scale(p)
    ft_put_item(p, b[i], getColor(0, flag=0), 12)



test()
root.mainloop()
