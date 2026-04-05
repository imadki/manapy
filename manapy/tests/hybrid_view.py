import tkinter as tk
import numpy as np
from manapy.domain import Domain, Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu, SingleCoreDomainTables
from manapy.backends.types import FLOAT_TYPE
from manapy.tests.meshes import get_mesh

########################################################
##### Create domain
########################################################

dim, mesh_path, mesh_name = get_mesh(1)


def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  partitioning = Partitioning(mesh)
  partitioning.make_n_part_mesh_nodal(nb_parts)
  local_domain_data = partitioning.create_sub_domains()

  local_domains = LocalDomain1Cpu.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, local_domains

size = 4
l_domains, l = create_domain(size)
g_domains, g = create_domain(1)

# ./create_test_domains
def _create_face_to_phyid(nb_faces, phyid_to_faceid: 'int32[:]'):
  face_to_phyid = np.ones(shape=nb_faces, dtype=np.int32) * -1
  face_to_phyid[phyid_to_faceid] = np.arange(phyid_to_faceid.shape[0])
  return face_to_phyid

# ./create_test_domains
def _remap_fid_to_phyid(cell_ghostnid, node_ghostid, face_to_phyid):
  for i in range(cell_ghostnid.shape[0]):
    cg = cell_ghostnid[i]
    for j in range(cg[-1]):
      fid = cg[j]
      cg[j] = face_to_phyid[fid]

  for i in range(node_ghostid.shape[0]):
    ng = node_ghostid[i]
    for j in range(ng[-1]):
      fid = ng[j]
      ng[j] = face_to_phyid[fid]

face_to_phyid = _create_face_to_phyid(len(l[0].faces), l[0].phyid_to_faceid)
_remap_fid_to_phyid(l[0].cell_ghostnid, l[0].node_ghostid, face_to_phyid)
########################################################
##### Render
########################################################

g_index = 0
g_2idx = 0

class   Renderer:
  def __init__(self, name):
    sc = 4
    self.root = tk.Tk()
    self.width = 1000 * sc
    self.height = 1500 * sc
    self.font_size = 5 * sc
    self.x_scale = 100 // 2.0 * sc
    self.x_offset = 200 // 2.0 * sc
    self.y_scale = 100 // 2.0 * sc
    self.y_offset = 100 // 2.0 * sc

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
      if g_index >= g_domains[0].nbcells:
        return
      g_index += 1

    def on_left_arrow(event):
      global g_index
      if g_index == 0:
        return
      g_index -= 1

    def on_w_key(event):
      global g_2idx
      if g_2idx >= 5:
        return
      g_2idx += 1

    def on_z_key(event):
      global g_2idx
      if g_2idx == 0:
        return
      g_2idx -= 1


    # Bind Right Arrow to the entire window
    self.root.bind("<Right>", on_right_arrow)
    self.root.bind("<Left>", on_left_arrow)
    self.root.bind("w", on_w_key)
    self.root.bind("z", on_z_key)

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
    a = self.get_rect_point(p, 0.3, 0.15)
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
    if i == g_index:
      ghosts = l[k].cell_ghostnid[i]
      for j in range(ghosts[-1]):
        ghost_info = l[k].ghost_info_flt[ghosts[j]]
        p = ghost_info
        render.ft_put_item(p, f"{p[0]:.3}, {p[1]:.3}", render.getColor(0))

def node_ghostid(k):
  for i in range(len(l[k].cells)):
    if i == g_index:
      cell = l[k].cells[i]
      for j in range(cell[-1]):
        if j == g_2idx:
          node_id = cell[j]
          render.ft_put_item(l[k].nodes[node_id], f"{j}", render.getColor(0))
          for z in range(l[k].node_ghostid[node_id, -1]):
            ghosts = l[k].node_ghostid[node_id]
            for y in range(ghosts[-1]):
              ghost_info = l[k].ghost_info_flt[ghosts[y]]
              p = ghost_info
              render.ft_put_item(p, f"{p[0]:.3}, {p[1]:.3}", render.getColor(0))

def tmp_1(k):
  for i in range(len(l[k].face_center)):
    p = l[k].face_center[i]
    render.ft_put_item(p, f"{i}", render.getColor(0))

def tmp_2(k):
  for i in range(len(l[k].cell_center)):
    p = l[k].cell_center[i]
    g_id = l[k].cell_loctoglob[i]
    render.ft_put_item(p, f"{g_id}", render.getColor(0))

def tmp_3(k):
  for i in range(len(l[k].cells)):
    for j in range(l[k].cells[i, -1]):
      node_id = l[k].cells[i][j]
      node_loc = l[k].node_loctoglob[node_id]
      p = l[k].nodes[node_id]
      render.ft_put_item(p, f"{node_loc}", render.getColor(0))

def test():
  for k in range(size):
    # Mesh
    for i in range(len(l[k].cells)):
      cell_nodeid = l[k].cells[i][0:l[k].cells[i, -1]]
      p = l[k].nodes[cell_nodeid][:, 0:2].flatten()
      render.create_polygon(p, render.getColor(k+1))
      if i == g_index:
        render.create_polygon(p, render.getColor(k))

    tmp_3(k)
    # draw_cellfid(k)
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
    # node_ghostid(k)


def redraw():
  FPS = 20
  FRAME_TIME = int(1000 / FPS)
  render.canvas.delete("all")
  test()
  render.root.after(FRAME_TIME, redraw)


redraw()
render.root.mainloop()