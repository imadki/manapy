import os
import numpy as np
from create_domain import Domain, Mesh

dim = 3
float_precision = 'float32'
root_file = os.getcwd()
mesh_path = 'tetrahedron.msh'
mesh_path = os.path.join(root_file, 'mesh', mesh_path)



mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
print(domain.cells.shape[0])


local_domains_data = domain.create_sub_domains(4)
print("====> End <=====")
print(local_domains_data)
