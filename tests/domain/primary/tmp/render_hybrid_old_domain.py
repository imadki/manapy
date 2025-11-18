import tkinter as tk
import numpy as np
from manapy.backends.types import FLOAT_TYPE
from manapy.tests.meshes import get_mesh
from manapy.tests.helpers.DomainTables import DomainTables


########################################################
##### Render
########################################################

class   Renderer:
  def __init__(self, name):
    self.root = tk.Tk()
    self.width = 1500
    self.height = 1000
    self.font_size = 12
    self.x_scale = 100
    self.x_offset = 100
    self.y_scale = 100
    self.y_offset = 100

    self.root.title(name)
    self.frame = tk.Frame(self.root)
    self.frame.pack(fill=tk.BOTH, expand=True)
    self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
    self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.canvas.configure(yscrollcommand=self.scrollbar.set)

    # Bind a callback to dynamically set the scroll region based on content size
    scrollregion_width = self.width
    scrollregion_height = self.height
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
    p[0::2] = p[0::2] * self.x_scale + self.x_offset
    p[1::2] = p[1::2] * self.y_scale + self.y_offset

  def get_font_size(self):
    return self.font_size

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
    a = self.get_rect_point(p, 0.2, 0.1)
    self.create_polygon(a, self.specialColor(1))

    self.scale(p)
    self.canvas.create_text(p[0], p[1], text=str(item), font=("Arial", self.get_font_size()),
                       fill=color)


render = Renderer("Hybrid")




def test():
  for k in range(size):
    # Mesh
    for i in range(len(l.d_cells[k])):
      cell_nodeid = l.d_cells[k][i][0:l.d_cells[k][i, -1]]
      p = l.d_nodes[k][cell_nodeid][:, 0:2].flatten()
      render.create_polygon(p, render.getColor(k))

    # # node global index
    # for i in range(len(l[k].nodes)):
    #   p = l[k].nodes[i]
    #   g_index = l[k].node_loctoglob[i] #d_node_loctoglob[i]
    #   render.ft_put_item(p, f"{g_index}", render.getColor(0))

    for i in range(len(l.d_cell_haloghostcenter[k])):
      p = l.d_cell_haloghostcenter[k][i][0:2]
      render.ft_put_item(p, f"{p}", render.getColor(k))

    # cell global index
    for i in range(len(l.d_cells[k])):
      g_index = l.d_cell_loctoglob[k][i]
      p = l.d_cell_center[k][i]
      render.ft_put_item(p, f"{k}, {g_index}", render.getColor(0))

    # for i in range(len(l.d_faces[k])):
    #   p = l.d_face_center[k][i]
    #   measure = l.d_face_measure[k][i]
    #   render.ft_put_item(p, f"{measure:.2}", render.getColor(0))

########################################################
##### Create domain
########################################################

dim, mesh_path, mesh_name = get_mesh(0)
size = 4
l = DomainTables(nb_partitions=size, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)
g = DomainTables(nb_partitions=size, mesh_name=mesh_name, float_precision=FLOAT_TYPE, dim=dim)

test()
render.root.mainloop()
