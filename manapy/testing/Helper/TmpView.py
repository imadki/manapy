import tkinter as tk
import numpy as np
import TriangleTables

########################################################
##### Create domain
########################################################

mesh = TriangleTables.TriangleTables()


########################################################
##### Render
########################################################

g_index = 0
g_2idx = 0

class   Renderer:
  def __init__(self, name):
    sc = 2
    self.root = tk.Tk()
    self.width = 1000 * sc
    self.height = 1500 * sc
    self.font_size = 10 * sc
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
      if g_index >= mesh.nb_cells:
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
    a = self.get_rect_point(p, 0.1, 0.15)
    self.create_polygon(a, self.specialColor(1))

    p = self.scale(p)
    self.canvas.create_text(p[0], p[1], text=str(item), font=("Arial", self.get_font_size()),
                       fill=color)


render = Renderer("Hybrid")



def test():
  for i in range(mesh.nb_cells):
    cell_nodeid = mesh.cells[i, 0:mesh.cells[i, -1]]
    p = mesh.nodes[cell_nodeid][:, 0:2].flatten()
    render.create_polygon(p, render.getColor(1))
    if i == g_index:
      render.create_polygon(p, render.getColor(2))

  # Start
  # for i in range(mesh.nb_cells):
  #   for j in range(mesh.cells[i, -1]):
  #     node_id = mesh.cells[i, j]
  #     if i == g_index:
  #       for k in range(mesh.node_ghostid[node_id, -1]):
  #         phy_id = mesh.node_ghostid[node_id, k]
  #         p = mesh.ghost_info_flt[phy_id]
  #         render.ft_put_item(p, f"{phy_id}", render.getColor(0))

  # node names
  print("=>", mesh.meshio_cells[119])
  for i in range(len(mesh.meshio_cells)):
    for j in range(mesh.meshio_cells[i, -1]):
      if i == 119:
        node_id = mesh.meshio_cells[i, j]
        p = mesh.meshio_nodes[node_id]
        print(p)
        render.ft_put_item(p, f"{node_id}", render.getColor(0))
        p = mesh.cell_center[i]
        render.ft_put_item(p, f"{i}", render.getColor(0))

  # faces name
  # for i in range(mesh.nb_cells):
  #   for j in range(mesh.cell_faceid[i, -1]):
  #     face_id = mesh.cell_faceid[i, j]
  #     p = mesh.face_center[face_id]
  #     render.ft_put_item(p, f"{mesh.face_oldname[face_id]}", render.getColor(0))



def redraw():
  FPS = 20
  FRAME_TIME = int(1000 / FPS)
  render.canvas.delete("all")
  test()
  render.root.after(FRAME_TIME, redraw)


redraw()
render.root.mainloop()