import tkinter as tk
import manapy.tests as test_folder
import os
import numpy as np
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.backends.types import FLOAT_TYPE


########################################################
##### Create domain
########################################################

mesh_list = [
  (2, 'hybrid.msh'),
  (2, 'rectangles.msh'),
  (2, 'triangles.msh')
]

root_file = os.path.dirname(os.path.abspath(test_folder.__file__))
dim, mesh_path = mesh_list[0] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'meshes', mesh_path) #tests/domain/primary/mesh


def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  local_domain_data = partitioning.create_sub_domains(nb_parts=nb_parts)

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, SingleCoreDomainTables(domains, FLOAT_TYPE), local_domains

size = 4
l_domains, domain_tables, l = create_domain(size)
g_domains, unified_domain, g = create_domain(1)


########################################################
##### Render
########################################################

class   Renderer:
  def __init__(self, name):
    print('Renderer')
    self.root = tk.Tk()

    self.root.title(name)
    self.frame = tk.Frame(self.root)
    self.frame.pack(fill=tk.BOTH, expand=True)
    self.canvas = tk.Canvas(self.root, width=3500, height=3000)
    self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.canvas.configure(yscrollcommand=self.scrollbar.set)

    # Bind a callback to dynamically set the scroll region based on content size
    scrollregion_width = 3500
    scrollregion_height = 3000
    def on_canvas_configure(event, canvas, scrollregion_width, scrollregion_height):
      canvas.configure(scrollregion=(0, 0, scrollregion_width, scrollregion_height))
    self.canvas.bind("<Configure>", lambda event: on_canvas_configure(event, self.canvas, scrollregion_width, scrollregion_height))
    self.colors = ["red", "blue", "green", "orange", "purple", "yellow", "magenta", "black", "gray",
            "white", "black"]


  def getColor(self, i):
    return self.colors[i % (len(self.colors)-2)]

  def specialColor(self, i):
    return self.colors[i % 2 + 9]

  def scale(self, p):
    p[:] = p * 100 + 200

  def get_font_size(self):
    return 24

  def create_polygon(self, points, color):
    self.scale(points)
    self.canvas.create_polygon(list(points), fill=color, outline="black", width=1)

  def get_rect_point(self, center, x, y):
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

  def ft_put_item(self, p, item, color):
    a = self.get_rect_point(p, 0.4, 0.2)
    self.create_polygon(a, self.specialColor(1))

    self.scale(p)
    self.canvas.create_text(p[0], p[1], text=str(item), font=("Arial", self.get_font_size()),
                       fill=color)


render = Renderer("Hybrid")




def test():
  for k in range(size):
    # Mesh
    for i in range(len(l[k].cells)):
      cell_nodeid = l[k].cells[i][0:l[k].cells[i, -1]]
      p = l[k].nodes[cell_nodeid][:, 0:2].flatten()
      render.create_polygon(p, render.getColor(k))

    # node global index
    for i in range(len(l[k].nodes)):
      p = l[k].nodes[i]
      g_index = l[k].node_loctoglob[i] #d_node_loctoglob[i]
      render.ft_put_item(p, f"{g_index}", render.getColor(0))

    # cell global index
    for i in range(len(l[k].cells)):
      g_index = l[k].cell_loctoglob[i]
      cell_type = l[k].cells_type[i]
      p = l[k].cell_center[i]
      render.ft_put_item(p, f"{cell_type}, {g_index}", render.getColor(0))

test()
render.root.mainloop()
