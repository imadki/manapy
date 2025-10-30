import os
from manapy.domain import Domain

mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'big/tetrahedron_big.msh'),
]
root_file = os.getcwd()
dim, mesh_path = mesh_list[4] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, '..', '..', 'tests', 'meshes', mesh_path) # manapy/tests


local_domain = Domain.create_domain(mesh_path, dim, recreate=True)



