import tkinter as tk
import numpy as np
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu
from manapy.backends.types import FLOAT_TYPE
from manapy.helpers import get_mesh

########################################################
##### Create domain
########################################################

dim, mesh_path, mesh_name = get_mesh(2)


def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  partitioning.make_n_part_mesh_nodal(nb_parts)
  local_domain_data = partitioning.create_sub_domains()

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  #domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return None, local_domains

size = 4
l_domains, l = create_domain(size)
g_domains, g = create_domain(1)


########################################################
##### Render
########################################################

g_index = 0

class   Renderer:
  def __init__(self, name):
    self.root = tk.Tk()
    self.width = 1500 * 2
    self.height = 1000 * 2
    self.font_size = 12 * 2
    self.x_scale = 100 // 2.0 * 2
    self.x_offset = 200 // 2.0 * 2
    self.y_scale = 100 // 2.0 * 2
    self.y_offset = 100 // 2.0 * 2

    self.root.title(name)
    self.frame = tk.Frame(self.root)
    self.frame.pack(fill=tk.BOTH, expand=True)
    self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
    self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.canvas.configure(yscrollcommand=self.scrollbar.set)

    def on_right_arrow(event):
      global g_index
      if g_index >= len(l[0].cells):
        return
      g_index += 1

    def on_left_arrow(event):
      global g_index
      if g_index == 0:
        return
      g_index -= 1

    # Bind Right Arrow to the entire window
    self.root.bind("<Right>", on_right_arrow)
    self.root.bind("<Left>", on_left_arrow)

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
    p = p.copy()
    p[0::2] = p[0::2] * self.x_scale + self.x_offset
    p[1::2] = p[1::2] * self.y_scale + self.y_offset
    return p

  def get_font_size(self):
    return self.font_size

  def create_polygon(self, points, color):
    points = self.scale(points)
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
    a = self.get_rect_point(p, 0.3, 0.3)
    self.create_polygon(a, self.specialColor(1))

    p = self.scale(p)
    self.canvas.create_text(p[0], p[1], text=str(item), font=("Arial", self.get_font_size()),
                       fill=color)


render = Renderer("Hybrid")


def draw_center(k):
  # cell->center
  for i in range(len(l[k].cells)):
    p = l[k].cell_center[i]
    center = l[k].cell_center[i]
    render.ft_put_item(p, f"{center[0]:.3}, {center[1]:.3}", render.getColor(0))

def draw_area(k):
  for i in range(len(l[k].cells)):
    p = l[k].cell_center[i]
    v = l[k].cell_volume[i]
    render.ft_put_item(p, f"{v}", render.getColor(0))

def draw_cellfid(k):
  # cell->cellfid
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell_cellfid = l[k].cell_cellfid[i]
      for j in range(cell_cellfid[-1]):
        p = l[k].cell_center[cell_cellfid[j]]
        render.ft_put_item(p, f"{j}", render.getColor(0))

def draw_cellnid(k):
  # cell->cellnid
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell_cellnid = l[k].cell_cellnid[i]
      for j in range(cell_cellnid[-1]):
        p = l[k].cell_center[cell_cellnid[j]]
        render.ft_put_item(p, f"{j}", render.getColor(0))

def draw_cellfaceid(k):
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell_faceid = l[k].cell_faceid[i]
      for j in range(cell_faceid[-1]):
        p = l[k].face_center[cell_faceid[j]]
        render.ft_put_item(p, f"{j}", render.getColor(0))

def draw_face_cellid(k):
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell_faceid = l[k].cell_faceid[i]
      for j in range(cell_faceid[-1]):
        face_id = cell_faceid[j]
        face_cellid = l[k].face_cellid[face_id]
        for z in face_cellid:
          if z != -1:
            p = l[k].cell_center[z]
            render.ft_put_item(p, f"{z}", render.getColor(0))

def draw_face_oldname(k):
  for i in range(l[k].faces.shape[0]):
    p = l[k].face_center[i]
    name = l[k].face_oldname[i]
    render.ft_put_item(p, f"{name}", render.getColor(0))

def draw_face_to_phyid(k):
  # we need just to check for phyid_to_faceid
  for i in range(len(l[k].phyid_to_faceid)):
    face_id = l[k].phyid_to_faceid[i]
    p = l[k].face_center[face_id]
    name = l[k].phy_faces_name[i]
    render.ft_put_item(p, f"{name}", render.getColor(0))

def draw_face_normal(k):
  for i in range(len(l[k].cells)):
    # if i == g_index:
    cell_faceid = l[k].cell_faceid[i]
    for j in range(cell_faceid[-1]):
      face_id = cell_faceid[j]
      p = l[k].face_center[face_id]
      normal = l[k].face_normal[face_id]
      render.ft_put_item(p, f"{normal[0]:.3}, {normal[1]:.3}", render.getColor(0))

def draw_face_measure(k):
  for i in range(len(l[k].cells)):
    # if i == g_index:
    cell_faceid = l[k].cell_faceid[i]
    for j in range(cell_faceid[-1]):
      face_id = cell_faceid[j]
      p = l[k].face_center[face_id]
      measure = l[k].face_measure[face_id]
      render.ft_put_item(p, f"{measure:.3}", render.getColor(0))

def draw_node_cellid(k):
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell_nodes = l[k].cells[i]
      for j in range(cell_nodes[-1]):
        node_id = cell_nodes[j]
        node_cellid = l[k].node_cellid[node_id]
        for z in range(node_cellid[-1]):
          item = node_cellid[z]
          p = l[k].cell_center[item]
          render.ft_put_item(p, f"{z}", render.getColor(0))

def draw_node_oldname(k):
  for i in range(l[k].nodes.shape[0]):
    p = l[k].nodes[i]
    name = l[k].node_oldname[i]
    render.ft_put_item(p, f"{name}", render.getColor(0))

def cell_ghostnid(k):
  for i in range(len(l[k].cells)):
    ghosts = l[k].cell_ghostnid[i]
    print(ghosts, l[k].shared_ghost_info.shape[0])
    for j in range(ghosts[-1]):
      ghost_id = l[k].face_to
      ghost_info = l[k].shared_ghost_info[ghosts[j]]
      render.ft_put_item(ghost_info, f"{j}", render.getColor(0))

def hhh(k):
  for i in range(len(l[k].ext_ghost_info_flt)):
    center = l[k].ext_ghost_info_flt[i, 0:3]
    render.ft_put_item(center, f"{i}", render.getColor(k+1))

def test():
  for k in range(size):
    # Mesh
    for i in range(len(l[k].cells)):
      cell_nodeid = l[k].cells[i][0:l[k].cells[i, -1]]
      p = l[k].nodes[cell_nodeid][:, 0:2].flatten()
      render.create_polygon(p, render.getColor(k+1))
      # if i == g_index:
      #   render.create_polygon(p, render.getColor(k))

    for i in range(len(l[k].cells)):
      cell_center = l[k].cell_center[i]
      gl =  l[k].cell_loctoglob[i]
      render.ft_put_item(cell_center, f"{k},{i}", render.getColor(0))
    hhh(k)
    # draw_center(k)
    # draw_area(k)
    # draw_cellfid(k)
    # draw_cellnid(k)
    # draw_cellfaceid(k)
    # draw_face_cellid(k)
    # draw_face_oldname(k)
    # draw_face_to_phyid(k)
    # draw_face_normal(k)
    # draw_face_measure(k)
    # draw_node_cellid(k)
    # draw_node_oldname(k)
    # cell_ghostnid(k)

    # # node global index
    # for i in range(len(l[k].nodes)):
    #   p = l[k].nodes[i]
    #   g_index = l[k].node_loctoglob[i] #d_node_loctoglob[i]
    #   render.ft_put_item(p, f"{g_index}", render.getColor(0))
    #
    # # cell global index
    # for i in range(len(l[k].cells)):
    #   g_index = l[k].cell_loctoglob[i]
    #   p = l[k].cell_center[i]
    #   render.ft_put_item(p, f"{k}, {g_index}", render.getColor(0))
    #
    # for i in range(len(l[k].cell_haloghostcenter)):
    #   p = l[k].cell_haloghostcenter[i][0:2]
    #   render.ft_put_item(p, f"{p}", render.getColor(k))

    # for i in range(len(l[k].faces)):
    #   p = l[k].face_center[i]
    #   measure = l[k].face_measure[i]
    #   render.ft_put_item(p, f"{measure:.2}", render.getColor(0))

# print(np.max(l[0].node_halophyid[:, 1]))

def redraw():
  FPS = 20
  FRAME_TIME = int(1000 / FPS)
  render.canvas.delete("all")
  test()
  render.root.after(FRAME_TIME, redraw)


redraw()
render.root.mainloop()