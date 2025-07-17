# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 render3d.py
from vedo import Text3D, show
import vedo
import sys
import os
import numpy as np


helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'domain')
sys.path.append(helpers_path)
from create_domain import Domain, Mesh, GlobalDomain, LocalDomain, SingleCoreDomainTables

mesh_list = [
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'cuboid.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
float_precision = 'float32' # the test does not support float64 or int64 yet
root_file = os.getcwd()
dim, mesh_path = mesh_list[1] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path) #tests/domain/primary/mesh

def create_domain(nb_parts):
  mesh = Mesh(mesh_path, dim)
  domain = GlobalDomain(mesh, float_precision)
  local_domain_data = domain.c_create_sub_domains(nb_parts)

  local_domains = LocalDomain.create_local_domains(local_domain_data)
  domains = [Domain(local_domains[i]) for i in range(len(local_domains))]

  return domains, SingleCoreDomainTables(domains, float_precision)

l_domains, domain_tables = create_domain(4)
#g_domains, unified_domain = create_domain(1)

domain = domain_tables
size = domain.nb_partitions

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

def render_cells():
  for k in range(size):
    loctoglob = domain.d_cell_loctoglob[k]
    for i in range(len(loctoglob)):
      g_index = domain.d_cell_loctoglob[k][i]
      c = domain.d_cell_faces[k][i]
      c = c[0:c[-1]]
      c = domain.d_face_oldname[k][c]
      #print(c)
      c = np.any(c != 0)
      if c:
        cell_center = domain.d_cell_center[k][i]
        p = cell_center
        cube = vedo.Box(pos=(p[0], p[1], p[2]),	length=1,	width=0.5,	height=1.5,	size=(),	c=colors[k],	alpha=1)
        objects.append(cube)
        #print(k, colors[k])

        cell_faces = domain.d_cell_faces[k][i]
        for j in range(cell_faces[-1]):
          face = domain.d_cell_faces[k][i, j]
          face_center = domain.d_face_center[k][face]
          face_oldname = domain.d_face_oldname[k][face]
          p = face_center
          #print(j, p, face_oldname)
          if face_oldname != 0:
            text = vedo.Text3D(str(g_index), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
            objects.append(text)

def render_node_name():
  for k in range(size):
    loctoglob = domain.d_cell_loctoglob[k]
    for i in range(len(loctoglob)):
      cell_center = domain.d_cell_center[k][i]
      p = cell_center
      #print(cell_center)
      cube = vedo.Box(pos=(p[0], p[1], p[2]), length=1, width=0.5, height=1.5, size=(), c='g4', alpha=0.5)
      #objects.append(cube)

      # cell_nodes = domain.d_cells[k][i]
      # for j in range(cell_nodes[-1]):
      #   nodeid = cell_nodes[j]
      #   node_vertices = domain_tables.d_nodes[k][nodeid]
      #   node_name = domain_tables.d_node_name[k][nodeid]
      #   p = node_vertices
      #   if node_name != 0 and node_name in [1, 2]:
      #     text = vedo.Text3D(str(node_name), pos=(p[0], p[1], p[2]), s=0.2, justify="center", c=colors[0])
      #     objects.append(text)

      cell_faces = domain.d_cell_faces[k][i]
      for j in range(cell_faces[-1]):
        face = domain.d_cell_faces[k][i, j]
        face_center = domain.d_face_center[k][face]
        face_oldname = domain.d_face_oldname[k][face]
        p = face_center
        if face_oldname != 0:
          #print(j, p, face_oldname)
          text = vedo.Text3D(str(face_oldname), pos=(p[0], p[1], p[2]), c=colors[k], s=0.2, justify="center")
          objects.append(text)

      # if i == 101:
      #   break


  # face_center = domain.d_face_center[k]
  # for i in range(len(face_center)):
  #   face_oldname = domain.d_face_oldname[k][i]
  #   p = face_center[i]
  #   text = vedo.Text3D(str(face_oldname), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
  #   objects.append(text)


#render_cells()
render_node_name()
show(*objects, axes=1, bg='white')
