# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 render3d.py
from vedo import Text3D, show
import vedo
import sys
import subprocess
import os
import numpy as np

import sys
import os
import numpy as np
import pymetis

# Add the parent directory to sys.path
helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(helpers_path)
from create_domain import Domain
from create_domain import Mesh


float_precision = 'float32'
dim=3
mesh_name = 'tetrahedron.msh'
root_file = os.getcwd()
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_name)
size = 1

mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
nb_parts = 4
local_domains = domain.create_sub_domains(nb_parts)

##########################################
##########################################
##########################################

# Create the cube centered at origin with side lengths of 1
objects = []

colors = [
  "tomato",
  "gold",
  "deepskyblue",
  "limegreen",
  "orchid",
  "orange",
  "dodgerblue",
  "salmon",
  "turquoise",
  "violet"
]

print(domain.phy_faces)

def part_render_node_name():
  for prt in range(nb_parts):
    phy_faces = local_domains[prt].phy_faces
    phy_faces_name = local_domains[prt].phy_faces_name
    nodes = local_domains[prt].nodes
    for i in range(len(phy_faces)):
      face = phy_faces[i]
      vertices = nodes[face[0:face[-1]]]
      p = np.sum(vertices, axis=0) / face[-1]
      color = colors[prt]
      text = vedo.Text3D(str(i), pos=(p[0], p[1], p[2]), s=0.2, justify="center", c=color)
      objects.append(text)


def render_node_name():
  phy_faces = domain.phy_faces
  phy_faces_name = domain.phy_faces_name
  node_phyfaceid = domain.node_phyfaceid
  faces = domain.faces
  face_name = domain.face_name
  nodes = domain.nodes
  i = 111
  for f in range(node_phyfaceid[i, -1]):
    face = phy_faces[node_phyfaceid[i, f]]
    for j in range(face[-1]):
      n = face[j]
      p = nodes[n]
      text = vedo.Text3D(str(n), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
      objects.append(text)

def render_cells():
  cells = domain.cells
  phy_faces = domain.phy_faces
  phy_faces_name = domain.phy_faces_name
  node_phyfaceid = domain.node_phyfaceid
  faces = domain.faces
  face_name = domain.face_name
  nodes = domain.nodes
  for i in range(cells.shape[0]):
    cell = cells[i]
    for j in range(cell[-1]):
      n = cell[j]
      if n in [110, 111, 362, 363, 597, 112, 589]:
        p = nodes[n]
        text = vedo.Text3D(str(i), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
        objects.append(text)

# render_cells()
# render_node_name()
part_render_node_name()
show(*objects, axes=1, bg='white')
