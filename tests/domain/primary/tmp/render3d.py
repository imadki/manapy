# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 render3d.py
from vedo import Text3D, show
import vedo
import sys
import subprocess
import os
import numpy as np
helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(helpers_path)
from helpers.DomainTables import DomainTables

mpi_exec = "/usr/bin/mpirun"
python_exec = "/home/aben-ham/anaconda3/envs/work/bin/python3"
float_precision = 'float32'
dim=3
mesh_name = 'cube.msh'


def create_partitions(nb_partitions, mesh_name, float_precision, dim):
  root_file = os.getcwd()
  mesh_file_path = os.path.join(root_file, '..', 'mesh', mesh_name)
  script_path = os.path.join(root_file, '..', 'helpers', 'create_partitions_mpi_worker.py')
  cmd = [mpi_exec, "-n", str(nb_partitions), "--oversubscribe", python_exec, script_path, mesh_file_path, float_precision, str(dim)]

  result = subprocess.run(cmd, env=os.environ.copy(), stderr=subprocess.PIPE)
  if result.returncode != 0:
    print(result.__str__(), os.getcwd())
    raise SystemExit(result.returncode)

domain_tables = DomainTables(nb_partitions=4, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
unified_domain = DomainTables(nb_partitions=1, mesh_name=mesh_name, float_precision=float_precision, dim=dim, create_par_fun=create_partitions)
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
      print(c)
      c = np.any(c != 0)
      if c:
        cell_center = domain.d_cell_center[k][i]
        p = cell_center
        cube = vedo.Box(pos=(p[0], p[1], p[2]),	length=1,	width=0.5,	height=1.5,	size=(),	c=colors[k],	alpha=1)
        objects.append(cube)
        print(k, colors[k])

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
      print(cell_center)
      cube = vedo.Box(pos=(p[0], p[1], p[2]), length=1, width=0.5, height=1.5, size=(), c='g4', alpha=0.5)
      #objects.append(cube)

      cell_nodes = domain.d_cells[k][i]
      for j in range(cell_nodes[-1]):
        nodeid = cell_nodes[j]
        node_vertices = domain_tables.d_nodes[k][nodeid]
        node_name = domain_tables.d_node_name[k][nodeid]
        p = node_vertices
        if node_name != 0:
          text = vedo.Text3D(str(node_name), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
          objects.append(text)
      # cell_faces = domain.d_cell_faces[k][i]
      # for j in range(cell_faces[-1]):
      #   face = domain.d_cell_faces[k][i, j]
      #   face_center = domain.d_face_center[k][face]
      #   face_oldname = domain.d_face_oldname[k][face]
      #   p = face_center
      #   # print(j, p, face_oldname)
      #   if face_oldname != 0:
      #     text = vedo.Text3D(str(face_oldname), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
      #     objects.append(text)

      # if i == 101:
      #   break


  # face_center = domain.d_face_center[k]
  # for i in range(len(face_center)):
  #   face_oldname = domain.d_face_oldname[k][i]
  #   p = face_center[i]
  #   text = vedo.Text3D(str(face_oldname), pos=(p[0], p[1], p[2]), s=0.2, justify="center")
  #   objects.append(text)


render_cells()
# render_node_name()
show(*objects, axes=1, bg='white')
