import os
from manapy.domain import Mesh, Partitioning
from manapy.tests import LocalDomain1Cpu
import h5py
import numpy as np



mesh_list = [
  (2, 'rectangles.msh'),
  (2, 'triangles.msh'),
  (2, 'hybrid.msh'),
  (3, 'cube.msh'),
  (3, 'tetrahedron.msh'),
  (3, 'tetrahedron_big.msh'),
]
root_file = os.getcwd()
dim, mesh_path = mesh_list[2] # also modify dim variable accordingly
mesh_path = os.path.join(root_file, 'meshes', mesh_path) #tests/domain/primary/mesh

# Create global domain
mesh = Mesh(mesh_path, dim)
partitioning = Partitioning(mesh)
local_domain_data = partitioning.create_sub_domains(nb_parts=1)
global_domain = LocalDomain1Cpu.create_local_domains(local_domain_data)


# Create the hd5 file


path = 'hybrid_test_tables.hd5'
with h5py.File(path, 'w') as f:
  f.create_dataset('cells', data=global_domain[0].cells)
  f.create_dataset('cell_center', data=global_domain[0].cell_center)
  f.create_dataset('cell_volume', data=global_domain[0].cell_volume)
  f.create_dataset('cell_cellfid', data=global_domain[0].cell_cellfid)
  f.create_dataset('cell_cellnid', data=global_domain[0].cell_cellnid)
  f.create_dataset('cell_faceid', data=global_domain[0].cell_faceid)
  f.create_dataset('faces', data=global_domain[0].faces)
  f.create_dataset('face_cellid', data=global_domain[0].face_cellid)
  f.create_dataset('face_center', data=global_domain[0].face_center)
  f.create_dataset('face_oldname', data=global_domain[0].face_oldname)
  f.create_dataset('phyid_to_faceid', data=global_domain[0].phyid_to_faceid)
  f.create_dataset('phy_faces', data=global_domain[0].phy_faces)
  f.create_dataset('face_normal', data=global_domain[0].face_normal)
  f.create_dataset('face_measure', data=global_domain[0].face_measure)
  f.create_dataset('face_tangent', data=global_domain[0].face_tangent)
  f.create_dataset('face_binormal', data=global_domain[0].face_binormal)
  f.create_dataset('nodes', data=global_domain[0].nodes)
  f.create_dataset('node_cellid', data=global_domain[0].node_cellid)
  f.create_dataset('node_oldname', data=global_domain[0].node_oldname)
  f.create_dataset('shared_ghost_info', data=global_domain[0].shared_ghost_info)
  f.create_dataset('cell_ghostnid', data=global_domain[0].cell_ghostnid)
  f.create_dataset('node_ghostid', data=global_domain[0].node_ghostid)