import os
from typing import Union
# gmsh ./meshes/tetra_test_2.geo -3 -setnumber Nx 100 -setnumber Ny 100 -setnumber Nz 100  -o ./meshes/big/tetra_test_100.msh > /dev/null

root_file = os.path.dirname(os.path.abspath(__file__))
meshes_folder = os.path.join(root_file, '..', '..', 'meshes')
test_meshes_folder = os.path.join(root_file, '..', '..', 'tests', 'data', 'meshes')

meshes_list = [
  # Meshes
  (2, f'{meshes_folder}/rectangles.msh'),
  (2, f'{meshes_folder}/triangles.msh'),
  (2, f'{meshes_folder}/hybrid2d.msh'),
  (3, f'{meshes_folder}/cuboid.msh'),
  (3, f'{meshes_folder}/tetrahedrons.msh'),
  (3, f'{meshes_folder}/hybrid3d.msh'),
  (2, f'{meshes_folder}/big/carre.msh'),
  (3, f'{meshes_folder}/big/tetra_test_100.msh'),
  (3, f'{meshes_folder}/big/tetra_test_200.msh'),
  (2, f'{meshes_folder}/big/carre1.msh'),
  # These are test meshes used for unit testing (don't renamed them)
  (2, f"{test_meshes_folder}/rectangles.msh"),
  (2, f"{test_meshes_folder}/triangles.msh"),
  (2, f"{test_meshes_folder}/smallTriangles.msh"),
  (2, f"{test_meshes_folder}/smallHybrid2D.msh"),
  (2, f"{test_meshes_folder}/hybrid2d.msh"),
  (3, f"{test_meshes_folder}/cuboid.msh"),
  (3, f"{test_meshes_folder}/smallCuboid.msh"),
  (3, f"{test_meshes_folder}/smallTetrahedrons.msh"),
  (3, f'{test_meshes_folder}/tetrahedrons.msh'),
  (3, f"{test_meshes_folder}/hybrid3d.msh"),
  (3, f"{test_meshes_folder}/smallHybrid3d.msh"),
]

"""
Get mesh with default dim using the list above
If mesh does not exist in the list is must be exist in the root_folder and dim must be specified
"""
def get_mesh_helper(root_folder: str, name: str, dim: int = None):
  dic = {}
  for item in meshes_list:
    dic[item[1]] = item[0]

  mesh_path = os.path.join(root_folder, name)
  if not os.path.isfile(mesh_path):
    raise ValueError("Mesh file does not exist")
  if mesh_path in dic:
    dim = dic[mesh_path]
    mesh_name = name
  else:
    if dim is None:
      raise ValueError("Dimension must be specified for mesh")
    mesh_name = name
  mesh_path = os.path.join(root_folder, mesh_name)
  return dim, mesh_path, mesh_name

def get_mesh(name: str, dim: int = None):
  return get_mesh_helper(meshes_folder, name, dim)

def get_test_mesh(name: str, dim: int = None):
  return get_mesh_helper(test_meshes_folder, name, dim)