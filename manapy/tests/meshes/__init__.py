import os

def get_mesh(name: 'int || str'):
  mesh_list = [
    (2, 'rectangles.msh'),
    (2, 'triangles.msh'),
    (2, 'hybrid.msh'),
    (2, 'carre.msh'),
    (3, 'cube.msh'),
    (3, 'tetrahedron.msh'),
    (3, 'tetrahedron_big.msh'),
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