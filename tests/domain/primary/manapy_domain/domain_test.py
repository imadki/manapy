import os
import numpy as np
import sys



helpers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(helpers_path)
from create_domain import Domain, Mesh

dim = 3
float_precision = 'float32'
root_file = os.getcwd()
mesh_path = 'tetrahedron_big.msh'
mesh_path = os.path.join(root_file, '..', 'mesh', mesh_path)



mesh = Mesh(mesh_path, dim)
domain = Domain(mesh, float_precision)
print(domain.cells.shape[0])


local_domains_data = domain.c_create_sub_domains(4)
print("====> End <=====")
#print(local_domains_data)
