import os

# gmsh ./meshes/tetra_test_2.geo -3 -setnumber Nx 100 -setnumber Ny 100 -setnumber Nz 100  -o ./meshes/big/tetra_test_100.msh > /dev/null

root_file = os.path.dirname(os.path.abspath(__file__))
meshes_folder = os.path.join(root_file, '..', '..', 'meshes')
test_meshes_folder = os.path.join(root_file, '..', '..', 'tests', 'data', 'meshes')

meshes_list = [
  (2, 'rectangles.msh'),#0
  (2, 'triangles.msh'),#1
  (2, 'hybrid2d.msh'),#2
  (3, 'cuboid.msh'),#3
  (3, 'tetrahedrons.msh'),#4
  (2, 'big/carre.msh'),#5
  (3, 'big/tetra_test_100.msh'),#6
  (3, 'big/tetra_test_200.msh'), #7
]

test_meshes_list = [
  (2, "rectangles.msh"),
  (2, "triangles.msh"),
  (2, "smallTriangles.msh"),
  (2, "smallHybrid2D.msh"),
  (2, "hybrid2d.msh"),
  (3, "cuboid.msh"),
  (3, "smallCuboid.msh"),
  (3, "smallTetrahedrons.msh"),
  (3, 'tetrahedrons.msh')
]

def get_mesh(name: 'int || str'):
  dic = {}
  for item in meshes_list:
    dic[item[1]] = item[0]
  if isinstance(name, int):
    dim, mesh_name = meshes_list[name]
  else:
    dim = dic[name]
    mesh_name = name
  mesh_path = os.path.join(meshes_folder, mesh_name)
  return dim, mesh_path, mesh_name

def get_test_mesh(name: 'int || str'):
  dic = {}
  for item in test_meshes_list:
    dic[item[1]] = item[0]
  if isinstance(name, int):
    dim, mesh_name = test_meshes_list[name]
  else:
    dim = dic[name]
    mesh_name = name
  mesh_path = os.path.join(test_meshes_folder, mesh_name)
  return dim, mesh_path, mesh_name