import os

def get_mesh(name: 'int || str'):
  mesh_list = [
    (2, 'rectangles.msh'),#0
    (2, 'triangles.msh'),#1
    (2, 'hybrid.msh'),#2
    (2, 'carre.msh'),#3
    (3, 'cube.msh'),#4
    (3, 'tetrahedron.msh'),#5
    # gmsh ./meshes/tetra_test_2.geo -3 -setnumber Nx 100 -setnumber Ny 100 -setnumber Nz 100  -o ./meshes/big/tetra_test_100.msh > /dev/null
    (3, 'big/tetra_test_100.msh'),#6
    (3, 'big/tetra_test_200.msh'),  #7
  ]
  dic = {}
  for item in mesh_list:
    dic[item[1]] = item[0]
  if isinstance(name, int):
    dim, mesh_name = mesh_list[name]
  else:
    dim = dic[name]
    mesh_name = name
  root_file = os.path.dirname(os.path.abspath(__file__))
  mesh_path = os.path.join(root_file, mesh_name)
  return dim, mesh_path, mesh_name